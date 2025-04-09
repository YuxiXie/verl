# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import regex
from typing import List
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.vllm_rollout import vLLMRollout, _pre_process_inputs
from verl.third_party.vllm import LLM

from verl.workers.rollout.vllm_rollout.mcts import (
    StepLMWorldModel, 
    MCTS, MCTSConfig, 
    StepLMConfig, SearchArgs,
    TreeConstructor,
)


class vLLMRolloutMCTS(vLLMRollout):

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__(actor_module, config, tokenizer, model_hf_config, **kwargs)
        
        self.mcts_depth_limit = 3
        world_model = StepLMWorldModel(max_length=1024, base_tokenizer=tokenizer)
        search_cfg = StepLMConfig(SearchArgs(
            base_tokenizer=tokenizer,
            sampling_params=self.sampling_params,
            n_actions=2,
            n_init_actions=3,
            depth_limit=self.mcts_depth_limit,
            force_terminating_on_depth_limit=True,
            include_demo=True,
            genlen_threshold=768,
            step_threshold=8,
        ))
        mcts_algo = MCTS(MCTSConfig(
            n_iters=128,
            depth_limit=8,
            breadth_limit=5,
        ))
        self.mcts_searcher = TreeConstructor(
            world_model=world_model, 
            search_config=search_cfg, 
            search_algo=mcts_algo,
        )
        
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']
        
        has_search = 'search_start_indices' in prompts.batch
        if has_search:
            search_start_index = prompts.batch['search_start_indices'][0]
            assert prompts.batch['search_start_indices'].eq(search_start_index).all(), "enforce left-padded"
            idx, attention_mask, position_ids = idx[:, :search_start_index], attention_mask[:, :search_start_index], position_ids[:, :search_start_index]

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }
        
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            output = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
            response = output[0].to(idx.device)
            # log_probs = output[1].to(idx.device)
        
        if is_validate:
            pass
        
        for index in range(len(idx_list)):
            demo = self.mcts_searcher.search_config.base_tokenizer.decode(response[index], skip_special_tokens=True)
            num_steps = len(regex.split(r'[\n\r]{2,}', demo))
            if num_steps > self.mcts_depth_limit:
                num_steps = min(num_steps, 32)
                self.mcts_searcher.search_config.depth_limit = num_steps
                self.mcts_searcher.search_algo.depth_limit = num_steps
                self.mcts_searcher.search_config.skip_sampling = (num_steps + self.mcts_depth_limit - 1) // self.mcts_depth_limit
            else:
                self.mcts_searcher.search_config.depth_limit = max(num_steps, self.mcts_depth_limit // 2)
                self.mcts_searcher.search_algo.depth_limit = self.mcts_depth_limit
                self.mcts_searcher.search_config.skip_sampling = 1
            
            mcts_rst = self.mcts_searcher(
                {'answer': prompts.non_tensor_batch['answer'][index], 'demo': prompts.non_tensor_batch['demo'][index]},
                policy_model=self.inference_engine,
                prompt_ids=idx_list[index],
            )
            import ipdb; ipdb.set_trace()
        
        mcts_rst = self.mcts_searcher(
            {'answer': gt_answer, 'demo': solution},
            policy_model=self.inference_engine,
            prompt_ids=idx_list[0],
            attention_mask=idx_list[0].ne(self.tokenizer.pad_token_id),
        )

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            output = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
            response = output[0].to(idx.device)
            # log_probs = output[1].to(idx.device)

            if response.shape[1] < self.config.response_length:
                response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
                # log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)
            
            if has_search:
                search_idx = prompts.batch['input_ids']
                search_attention_mask = prompts.batch['attention_mask']
                search_position_ids = prompts.batch['position_ids']
                if self.sampling_params.n > 1 and do_sample:
                    search_idx = search_idx.repeat_interleave(self.sampling_params.n, dim=0)
                    search_attention_mask = search_attention_mask.repeat_interleave(self.sampling_params.n, dim=0)
                    search_position_ids = search_position_ids.repeat_interleave(self.sampling_params.n, dim=0)
                    search_start_indices = prompts.batch['search_start_indices'].repeat_interleave(self.sampling_params.n, dim=0)
                    search_masks = prompts.batch['search_masks'].repeat_interleave(self.sampling_params.n, dim=0)

            # utilize current sampling params
            if self.sampling_params.n > 1 and do_sample:
                idx = idx.repeat_interleave(self.sampling_params.n, dim=0)
                attention_mask = attention_mask.repeat_interleave(self.sampling_params.n, dim=0)
                position_ids = position_ids.repeat_interleave(self.sampling_params.n, dim=0)
                batch_size = batch_size * self.sampling_params.n
            seq = torch.cat([idx, response], dim=-1)
            if has_search:
                search_seq = torch.cat([search_idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response,
                                                    eos_token=eos_token_id,
                                                    dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        if has_search:
            search_position_ids = torch.cat([search_position_ids, search_position_ids[:, -1:] + delta_position_id], dim=-1)
            search_attention_mask = torch.cat((search_attention_mask, response_attention_mask), dim=-1)
            search_masks = torch.cat((search_masks, torch.zeros_like(response_attention_mask.bool())), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch_dict = {
            'prompts': idx,
            'responses': response,
            'input_ids': seq,  # here input_ids become the whole sentences
            # 'old_log_probs': log_probs, # we will recompute old log prob with actor
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }
        if has_search:
            batch_dict.update({
                'search_input_ids': search_seq,
                'search_attention_mask': search_attention_mask,
                'search_position_ids': search_position_ids,
                'search_start_indices': search_start_indices,
                'search_masks': search_masks,
            })
        batch = TensorDict(
            batch_dict,
            batch_size=batch_size)

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)

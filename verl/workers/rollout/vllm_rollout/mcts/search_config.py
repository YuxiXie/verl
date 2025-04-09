# Reference: https://github.com/maitrix-org/llm-reasoners/blob/main/examples/RAP/gsm8k/search_config.py

from typing import NamedTuple

import math
import regex
import random
import numpy as np
from tqdm import trange, tqdm
from typing import Tuple
from contextlib import contextmanager

from transformers import PreTrainedTokenizerBase

from vllm import SamplingParams

from verl.third_party.vllm import LLM
from verl.utils.reward_score.math_eval import extract_answer, math_equal
from verl.workers.rollout.vllm_rollout.mcts.base import SearchConfig
from verl.workers.rollout.vllm_rollout.mcts.world_model import StepLMState, StepLMAction


class SearchArgs(NamedTuple):
    base_tokenizer: PreTrainedTokenizerBase
    sampling_params: SamplingParams = None
    
    n_actions: int = 8
    n_init_actions: int = 16
    depth_limit: int = 16
    force_terminating_on_depth_limit: bool = False
    
    include_demo: bool = False
    
    genlen_threshold: int = 512
    step_threshold: int = 10


class StepLMConfig(SearchConfig):
    def __init__(self, args: SearchArgs) -> None:
        super().__init__()
        self.example = None
        self.include_demo = args.include_demo
        
        self.skip_sampling = 1
        
        self.n_actions = args.n_actions
        self.n_init_actions = args.n_init_actions
        self.force_terminating_on_depth_limit = args.force_terminating_on_depth_limit
        self.depth_limit = args.depth_limit
        
        self.base_tokenizer = args.base_tokenizer
        self.sampling_params = args.sampling_params
        
        self.genlen_threshold = args.genlen_threshold
        self.step_threshold = args.step_threshold

    def _get_sequence_ids_in_path(self, state: StepLMState):
        steps = [s.next_step_ids for s in state]
        return [token for step in steps for token in step], [token for step in steps[1:] for token in step]

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)
    
    def get_actions(self, policy_model: LLM, state: StepLMState, existed_actions: list=[]) -> list[StepLMAction]:
        at_depth_limit = self.force_terminating_on_depth_limit and len(state) >= self.depth_limit
        
        n_actions = self.n_init_actions if not len(state) else self.n_actions
        if at_depth_limit or (len(existed_actions) and self.include_demo) or (len(state) - 1) % self.skip_sampling != 0:
            # 1) reach depth limit; 2) already include demo; 3) skip sampling
            n_actions = 1
        
        input_ids, response_ids = self._get_sequence_ids_in_path(state)
        
        # set terminators / stop-strings
        terminators = [self.base_tokenizer.eos_token_id, self.base_tokenizer.pad_token_id]
        stop_strings = [self.base_tokenizer.eos_token, self.base_tokenizer.pad_token]
        terminators, stop_strings = [x for x in terminators if x is not None], [x for x in stop_strings if x is not None]
        kwargs = {
            'n': max(n_actions - len(existed_actions), 1), 
            'temperature': 1.0, 
            'top_p': 0.95,
            'repetition_penalty': 1.25,
            'max_tokens': max(1, self.sampling_params.max_tokens - len(response_ids)), 
            'stop': None, 
            'detokenize': False,
        }
        if not at_depth_limit:
            stop_strings += ["\n\n"]
            kwargs.update({
                'stop': list(set(stop_strings)),
                'detokenize': True,
            })
        
        # ininitalize actions
        action_ids, unique_sequences = [], []
        if len(existed_actions):
            unique_sequences.extend(self.base_tokenizer.batch_decode(existed_actions))
        
        # include demo
        if self.include_demo and self.example['demo'] is not None:
            generated = self.base_tokenizer.decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            demo_steps = regex.split(r'[\n\r]{2,}', self.example['demo'].strip())
            if len(demo_steps) > self.depth_limit:
                demo_steps = demo_steps[:self.depth_limit] + ['\n\n'.join(demo_steps[self.depth_limit - 1:])]
            
            cur_step = ''
            for sid, step in enumerate(demo_steps):
                step = f'{step}\n\n' if sid < len(demo_steps) - 1 else step
                if generated.lstrip() == cur_step.lstrip():
                    if step not in unique_sequences:
                        gen_ids = self.base_tokenizer(
                            step, add_special_tokens=False, return_tensors='pt',
                        )['input_ids'][0].tolist()
                        
                        action_ids.append(gen_ids)
                        unique_sequences.append(step)
                    break
                cur_step += step
        
        # generation
        with self.update_sampling_params(**kwargs):
            output_ids, output_logprobs = policy_model.generate(
                prompts=None,
                sampling_params=self.sampling_params,
                prompt_token_ids=input_ids,
                use_tqdm=False,
            )
        
        output_sequences = self.base_tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
        for gen_ids, seq in zip(output_ids, output_sequences):
            if seq in unique_sequences:
                continue
            gen_ids = gen_ids[gen_ids.ne(self.base_tokenizer.pad_token_id).nonzero().squeeze(-1)].cpu().tolist()
            action_ids.append(gen_ids)
            unique_sequences.append(seq)
        
        return action_ids

    def get_values(self, state: StepLMState, actions: list[StepLMAction]) -> list[tuple[float, bool]]:
        _, response_ids = self._get_sequence_ids_in_path(state)
        generated = self.base_tokenizer.decode(response_ids, skip_special_tokens=False)
        candidate_steps = self.base_tokenizer.batch_decode(actions, skip_special_tokens=False)
        
        results = []
        for step, action in zip(candidate_steps, actions):
            is_terminal = step.endswith(self.base_tokenizer.eos_token) or not step.endswith('\n')
            correctness = 0.0
            if is_terminal:
                is_correct = math_equal(extract_answer(generated + step), self.example['answer'])
                correctness = 1.0 if is_correct else -1.0
            genlen = len(response_ids) + len(action)
            n_steps = len(regex.split(r'[\n\r]{2,}', (generated + step).strip()))
            
            # apply length/depth penalties
            if genlen > self.genlen_threshold or n_steps > self.step_threshold:
                if correctness != 0:
                    length_score = (math.exp(min(0, self.genlen_threshold - genlen) / self.genlen_threshold) \
                        * math.exp(min(0, self.step_threshold - n_steps) / self.step_threshold)) ** 0.5
                    correctness = length_score if correctness > 0 else -length_score
            
            results.append((correctness, is_terminal))
        
        return results

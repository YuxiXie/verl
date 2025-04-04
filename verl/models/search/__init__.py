"""Auto-models for search models."""

from __future__ import annotations

import functools
import importlib
from dataclasses import dataclass
from collections import OrderedDict
from typing import Any, List, Union

import torch

from transformers import (
    AutoModelForCausalLM,
    TextIteratorStreamer, 
    DynamicCache, 
    AutoTokenizer, TextStreamer,
    LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper,
    StoppingCriteriaList, EosTokenCriteria, MaxLengthCriteria, StopStringCriteria, 
)
from transformers.utils import ModelOutput

import transformers.models.auto as auto_module
from transformers.models.auto.auto_factory import (
    _BaseAutoModelClass,
    _LazyAutoMapping,
    auto_class_update,
    getattribute_from_module,
)
from transformers.models.auto.configuration_auto import (
    CONFIG_MAPPING_NAMES,
    model_type_to_module_name,
)


class _LazyAutoMappingInSearch(_LazyAutoMapping):
    def _load_attr_from_module(self, model_type: str, attr: str) -> Any:
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(
                f'.{module_name}',
                'verl.models.search',
            )
        return getattribute_from_module(self._modules[module_name], attr)


MODEL_FOR_SEARCH_MAPPING_NAMES: OrderedDict[str, str] = OrderedDict(
    [
        # Search model mapping
        # ('llama', 'LlamaModelForSearch'),
        ('qwen2', 'Qwen2ModelForSearch'),
    ],
)
MODEL_FOR_SEARCH_MAPPING: OrderedDict[str, Any] = _LazyAutoMappingInSearch(
    CONFIG_MAPPING_NAMES,
    MODEL_FOR_SEARCH_MAPPING_NAMES,
)


@functools.partial(auto_class_update, head_doc='search model')
class AutoModelForSearch(_BaseAutoModelClass):
    _model_mapping: OrderedDict[str, Any] = MODEL_FOR_SEARCH_MAPPING


setattr(auto_module, 'MODEL_FOR_SEARCH_MAPPING', MODEL_FOR_SEARCH_MAPPING)  # noqa: B010
setattr(auto_module, AutoModelForSearch.__name__, AutoModelForSearch)


@dataclass
class SearchModelOutput(ModelOutput):
    """
    Output of the score model.

    Args:
        rewards (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Prediction scores of the score model.
        step_rewards (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_steps)`):
            Prediction scores of the score model.
        end_rewards (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,)`):
            Prediction scores of the end of the sequence.
    """

    rewards: torch.Tensor | None = None  # size = (B, L)
    step_rewards: torch.Tensor | None = None  # size = (B, T)
    end_rewards: torch.Tensor | None = None  # size = (B,)
    hidden_states: torch.Tensor | None = None  # size = (B,)


class SearchModelMixin:
    """Base class for search models."""

    n_search: int
    search_token_type: str

    def init_search(
        self, 
        n_search: int,
        search_token_type: str = 'identical',
    ) -> None:
        self.n_search = n_search
        self.search_token_type = search_token_type

    def refine_generate(
        self,
        input_ids: torch.LongTensor,    # (B, L)
        attention_mask: torch.BoolTensor | None = None,     # (B, L)
        search_hidden_states: torch.Tensor | None = None,
        max_length: int | None = 1024,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_strings: Union[str, List[str]] | None = None,
        verbal: bool = False,
        use_cache: bool = True,
        return_search_hidden_states: bool = False,
    ):
        # Validate the `.generate()` call
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        assert batch_size == 1, "Only support batch size 1 for now !!!"
        assert max_length > seq_length, "Input sequence length exceeds maximum length !!!"
        return_search_hidden_states = return_search_hidden_states and self.num_search_states > 0
        
        # Prepare logits processor
        processors = LogitsProcessorList()
        if do_sample:
            if temperature is not None and temperature > 0 and temperature != 1:
                processors.append(TemperatureLogitsWarper(temperature))
            elif top_p is not None:
                processors.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))
        
        # Prepare stopping criteria
        stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(EosTokenCriteria(eos_token_id=self.tokenizer.eos_token_id))
        if max_length is not None:
            max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
            stopping_criteria.append(MaxLengthCriteria( max_length=max_length, max_position_embeddings=max_position_embeddings))
        if stop_strings is not None:
            stopping_criteria.append(StopStringCriteria(stop_strings=stop_strings, tokenizer=self.tokenizer))
        
        # Prepare streamer
        if verbal:
            streamer = TextIteratorStreamer(self.tokenizer)
            streamer.put(input_ids.cpu())
        
        # Prepare inputs
        inputs_embeds = None
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        inputs = {
            'input_ids': None,
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'output_attentions': self.num_search_states > 0,
            'use_cache': use_cache,
            'output_hidden_states': self.num_search_states > 0 or use_cache,
        }
        
        # Papare the cache
        cache_name = "past_key_values"
        if use_cache:
            inputs[cache_name] = DynamicCache()
        inputs['cache_position'] = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1
        
        # Prepare search states
        if self.num_search_states > 0:
            residual = None
            search_start, search_end, _ = self._locate_search_tokens(input_ids[0])
        
        # Token-by-token generation
        keep_generate,  = True, 
        while keep_generate:
            # prepare bi-directional attention mask
            inputs['attention_mask'] = attention_mask
            if self.num_search_states > 0:
                inputs['attention_mask'] = self._prepare_mask(
                    input_ids=input_ids[0],
                    attention_mask=attention_mask[0],
                    only_among_search_states=True,
                ).unsqueeze(0).unsqueeze(1)
            
            # prepare input embs
            inputs_embeds = self.model.embed_tokens(input_ids) if inputs_embeds is None else torch.cat(
                [inputs_embeds, self.model.embed_tokens(input_ids[:, -1:])], dim=1
            )
            if self.num_search_states > 0:
                if residual is None:
                    residual = inputs_embeds[0, search_start: search_end + 1]
                if search_hidden_states is not None:
                    inputs_embeds[0, search_start: search_end + 1] = residual + search_hidden_states
            inputs['inputs_embeds'] = inputs_embeds
            
            # create missing `position_ids` on the fly
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            
            # slice model inputs if it's an input that should have the same length as `input_ids`
            if inputs[cache_name] is not None:
                position_ids = position_ids[:, -input_ids.shape[1]:]
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)
                inputs["position_ids"] = position_ids
                
                if inputs['inputs_embeds'].shape[1] != inputs['cache_position'].shape[0]:
                    inputs['inputs_embeds'] = inputs['inputs_embeds'][:, inputs['cache_position'], :]
                    inputs['position_ids'] = inputs['position_ids'][:, inputs['cache_position']]
                    if inputs['attention_mask'].dim() == 2:
                        inputs['attention_mask'] = inputs['attention_mask'][:, inputs['cache_position']]
                    else:
                        inputs['attention_mask'] = inputs['attention_mask'][:, :, inputs['cache_position'], :]
            
            # forward pass to get next token
            outputs = self(**inputs, return_dict=True)
            
            # update attention mask
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
            # update search hidden states
            if self.num_search_states > 0 and search_hidden_states is None:
                search_hidden_states = outputs.hidden_states[-1].clone()[0, search_start: search_end + 1]
        
            # update past_key_values keeping its naming used in model code
            cache_name, cache = self._extract_past_from_model_output(outputs)
            inputs[cache_name] = cache
            # update cache position
            if use_cache:
                inputs['cache_position'] = inputs['cache_position'][-1:] + 1
            else:
                past_positions = inputs.pop('cache_position')
                new_positions = torch.arange(
                    past_positions[-1] + 1, past_positions[-1] + 1 + 1, dtype=past_positions.dtype, device=past_positions.device,
                )
                inputs['cache_position'] = torch.cat((past_positions, new_positions))
            
            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits.clone()[:, -1, :]    #.float()
            # pre-process distribution
            next_token_scores = processors(input_ids, next_token_logits)    # (1, V)
            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)  # (1, V)
                # sample and get candidate tokens
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)   # (1, 1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)    # (1, 1)
            
            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if verbal:
                streamer.put(next_tokens.cpu())
                print(self.tokenizer.decode(input_ids[0]))
            
            unfinished_sequences = ~stopping_criteria(input_ids, next_token_scores)
            keep_generate = unfinished_sequences.max() != 0

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs
        
        if verbal: streamer.end()

        # Conduct self-rewarding
        # prepare labels
        labels = input_ids.clone()
        labels[:, :seq_length] = IGNORE_INDEX
        # calculate rewards
        rewards_outputs = self.get_reward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_step_rewards=True,
            return_dict=True,
            output_hidden_states=return_search_hidden_states,
        )
        if return_search_hidden_states:
            rewards_outputs.hidden_states = rewards_outputs.hidden_states[0, search_start: search_end + 1]
        
        return input_ids[:, seq_length:], rewards_outputs

    def iterative_refine_generate(
        self,
        input_ids: torch.LongTensor,    # (B, L)
        num_iteration: int = 1,
        step_by_step: bool = False,
        max_length: int | None = 1024,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        verbal: bool = True,
        use_cache: bool = True,
    ):
        # Validate the `.generate()` call
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        assert batch_size == 1, "Only support batch size 1 for now !!!"
        assert max_length > seq_length, "Input sequence length exceeds maximum length !!!"
        
        # Prepare stop strings
        stop_strings = None
        if step_by_step:
            stop_strings = ['\n\n']
        
        keep_generate, outputs_track, cur_iter = True, [], 1
        search_hidden_states = None
        while keep_generate:
            # refine generation
            gen_ids, rewards_outputs = self.refine_generate(
                input_ids=input_ids,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                stop_strings=stop_strings,
                verbal=verbal,
                use_cache=use_cache,
                search_hidden_states=search_hidden_states,
                return_search_hidden_states=num_iteration > 1 or step_by_step,
            )

            # update inputs
            input_ids = torch.cat([input_ids, gen_ids], dim=1)
            if num_iteration > 1 or step_by_step:
                search_hidden_states = rewards_outputs.hidden_states
            
            # update outputs
            outputs_track.append({
                'iteration': cur_iter,
                'new_tokens_num': gen_ids.size(-1),
                'gen_ids': input_ids[0, seq_length:].cpu(),
                'rewards': rewards_outputs.rewards.cpu()[0],
                'step_rewards': rewards_outputs.step_rewards.cpu()[0],
                'end_reward': rewards_outputs.end_rewards.cpu(),
            })
            
            # update iteration hyperparameters
            # max length limit
            if input_ids.size(-1) >= max_length:
                input_ids = input_ids[:, :seq_length]
                cur_iter += 1
            # finish a complete response
            elif gen_ids[0, -1].eq(self.tokenizer.eos_token_id):
                input_ids = input_ids[:, :seq_length]
                cur_iter += 1
            # check iteration limit
            elif not step_by_step:
                input_ids = input_ids[:, :seq_length]
                cur_iter += 1
                
            if cur_iter > num_iteration:
                keep_generate = False
            
        return outputs_track
    
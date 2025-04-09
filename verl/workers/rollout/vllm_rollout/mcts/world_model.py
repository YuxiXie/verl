# Adapted from: https://github.com/maitrix-org/llm-reasoners/blob/main/examples/RAP/gsm8k/world_model.py

from typing import NamedTuple, TypedDict

from transformers import GenerationConfig, PreTrainedTokenizerBase

from verl.workers.rollout.vllm_rollout.mcts.base import WorldModel

StepLMAction = list[int]

class StepSubResult(NamedTuple):
    next_step_ids: StepLMAction

StepLMState = list[StepSubResult]

class WorldModelArgs(NamedTuple):
    base_tokenizer: PreTrainedTokenizerBase
    stop_tokens: list[str] = []

class LMExample(TypedDict):
    input_ids: StepLMAction     # (L,)

class StepLMWorldModel(WorldModel[StepLMState, StepLMAction, LMExample]):
    def __init__(
        self,
        max_length: int,
        base_tokenizer: PreTrainedTokenizerBase,
        stop_tokens=[],
    ) -> None:
        super().__init__()
        self.base_tokenizer = base_tokenizer
        self.max_tokens_num = max_length
        self.stop_tokens = list(set(
            stop_tokens + [self.base_tokenizer.eos_token, self.base_tokenizer.pad_token]
        ))

    def is_terminal(self, state: StepLMState) -> bool:
        sum_tokens_num = sum(len(x.next_step_ids) for x in state[1:])
        
        if sum_tokens_num >= self.max_tokens_num:
            return True
        elif state[-1].next_step_ids[-1] in [self.base_tokenizer.eos_token_id, self.base_tokenizer.pad_token_id]:
            return True
        else:
            return False

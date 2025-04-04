from __future__ import annotations

from typing import Any

from transformers import Qwen2ForCausalLM
from transformers.configuration_utils import PretrainedConfig


from verl.models.search import SearchModelMixin

class Qwen2ModelForSearch(SearchModelMixin, Qwen2ForCausalLM):

    def __init__(self, config: PretrainedConfig, **kwargs: Any) -> None:
        super().__init__(config)
        self.model_type = 'qwen2'
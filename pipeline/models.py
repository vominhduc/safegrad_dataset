"""LLM and VLM inference wrappers for the SafeGrad pipeline.

Provides two main classes:

  LocalChatModel
    Async HuggingFace causal-LM wrapper with dynamic request batching.
    Used in Stage 2 for severity judging and prompt interpolation.

  LocalVisionModel
    Async HuggingFace vision-language model wrapper with dynamic batching.
    Used in Stage 4 for visual monotonicity verification.

Both classes coalesce concurrent ``complete()`` calls into single GPU batch
calls for efficient throughput.

Factory functions
-----------------
  get_local_chat_model(model_name, max_batch_size=8) -> LocalChatModel
  get_local_vision_model(model_name, max_batch_size=4) -> LocalVisionModel

HuggingFace token resolution (for gated models like Llama-3)
-------------------------------------------------------------
  resolve_hf_token() -> str | None
    Reads from HF_TOKEN / HUGGINGFACE_HUB_TOKEN env vars or
    ~/.cache/huggingface/token.
"""

from pipeline.local_llm import LocalChatModel, get_local_chat_model  # noqa: F401
from pipeline.local_vlm import LocalVisionModel, get_local_vision_model  # noqa: F401
from pipeline.hf_auth import resolve_hf_token  # noqa: F401

__all__ = [
    "LocalChatModel",
    "get_local_chat_model",
    "LocalVisionModel",
    "get_local_vision_model",
    "resolve_hf_token",
]

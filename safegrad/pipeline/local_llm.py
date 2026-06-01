from __future__ import annotations

import asyncio
import gc
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# How long (seconds) to wait for more requests before flushing a partial batch.
_BATCH_TIMEOUT = 0.02
# Worker exits after this many seconds of idle to free resources.
_WORKER_IDLE_TIMEOUT = 10.0


@dataclass
class _Request:
    system_prompt: str
    user_prompt: str
    temperature: float
    max_new_tokens: int
    future: asyncio.Future = field(compare=False)


class LocalChatModel:
    """Local HuggingFace causal-LM wrapper with dynamic request batching.

    Multiple concurrent ``complete()`` callers are coalesced into a single
    ``model.generate()`` call on the GPU, dramatically improving throughput
    compared to the previous one-request-at-a-time approach.

    Parameters
    ----------
    max_batch_size:
        Maximum number of prompts processed in one ``model.generate()`` call.
        Higher values use more VRAM but yield better GPU utilisation.
    """

    def __init__(self, model_name: str, max_batch_size: int = 8) -> None:
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self._load_lock = asyncio.Lock()
        self._tokenizer = None
        self._model = None
        # Queue and worker are created lazily inside the running event loop.
        self._queue: asyncio.Queue[_Request] | None = None
        self._worker_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    async def _ensure_loaded(self):
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model
        async with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return self._tokenizer, self._model
            self._tokenizer, self._model = await asyncio.to_thread(self._load_sync)
            return self._tokenizer, self._model

    def _load_sync(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("Loading local chat model: %s", self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        load_kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
            load_kwargs["torch_dtype"] = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if getattr(model.generation_config, "pad_token_id", None) is None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        log.info("Local chat model ready: %s", self.model_name)
        return tokenizer, model

    # ------------------------------------------------------------------
    # Batched generation
    # ------------------------------------------------------------------

    @staticmethod
    def _device_for(model) -> str:
        device = getattr(model, "device", None)
        if device is not None and str(device) != "meta":
            return str(device)
        try:
            return str(next(model.parameters()).device)
        except StopIteration:
            return "cpu"

    @staticmethod
    def _format_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return (
            f"System:\n{system_prompt}\n\n"
            f"User:\n{user_prompt}\n\n"
            "Assistant:\n"
        )

    def _generate_batch_sync(
        self,
        tokenizer,
        model,
        batch: list[_Request],
    ) -> list[str]:
        """Run a single model.generate() over all requests in *batch*.

        Left-padding is used (standard for decoder-only causal LM batch
        inference) so that each sequence ends at the same position and
        the model can generate without position-index confusion.
        """
        import torch

        prompts = [
            self._format_prompt(tokenizer, req.system_prompt, req.user_prompt)
            for req in batch
        ]
        max_new_tokens = max(req.max_new_tokens for req in batch)
        temperature = batch[0].temperature  # all phase calls use temperature=0

        orig_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        finally:
            tokenizer.padding_side = orig_padding_side

        device = self._device_for(model)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        padded_len = inputs["input_ids"].shape[1]

        generate_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)

        results: list[str] = []
        for seq in outputs:
            new_tokens = seq[padded_len:]
            results.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return results

    # ------------------------------------------------------------------
    # Background worker: collects requests, forms batches, runs generate
    # ------------------------------------------------------------------

    def _get_queue(self) -> asyncio.Queue[_Request]:
        """Return the queue for the *current* event loop, recreating if needed."""
        loop = asyncio.get_running_loop()
        if self._queue is None or self._loop is not loop:
            self._queue = asyncio.Queue()
            self._worker_task = None
            self._loop = loop
        return self._queue

    async def _ensure_worker(self) -> None:
        self._get_queue()  # ensure queue exists for current loop
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._batch_worker())

    async def _batch_worker(self) -> None:
        queue = self._get_queue()
        tokenizer, model = await self._ensure_loaded()

        while True:
            # Block until the first request arrives (or idle timeout).
            try:
                first = await asyncio.wait_for(queue.get(), timeout=_WORKER_IDLE_TIMEOUT)
            except asyncio.TimeoutError:
                # No work for a while — exit so the task doesn't linger.
                log.debug("Batch worker idle timeout, exiting.")
                return

            batch: list[_Request] = [first]

            # Collect additional requests that arrive within the batching window.
            deadline = asyncio.get_event_loop().time() + _BATCH_TIMEOUT
            while len(batch) < self.max_batch_size:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(queue.get(), timeout=remaining)
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

            log.debug("Flushing batch of %d request(s).", len(batch))
            try:
                results = await asyncio.to_thread(
                    self._generate_batch_sync, tokenizer, model, batch
                )
                for req, result in zip(batch, results):
                    if not req.future.done():
                        req.future.set_result(result)
            except Exception as exc:
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> str:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        req = _Request(system_prompt, user_prompt, temperature, max_new_tokens, fut)
        queue = self._get_queue()
        await self._ensure_worker()
        await queue.put(req)
        return await fut

    def unload(self) -> None:
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
            self._worker_task = None
        if self._model is None:
            return
        try:
            if hasattr(self._model, "to"):
                self._model.to("cpu")
        except Exception:
            pass
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_LOCAL_MODEL_CACHE: dict[str, LocalChatModel] = {}


def get_local_chat_model(model_name: str, max_batch_size: int = 8) -> LocalChatModel:
    if model_name not in _LOCAL_MODEL_CACHE:
        _LOCAL_MODEL_CACHE[model_name] = LocalChatModel(model_name, max_batch_size)
    return _LOCAL_MODEL_CACHE[model_name]

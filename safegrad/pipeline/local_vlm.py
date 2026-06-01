from __future__ import annotations

import asyncio
import base64
import gc
import io
import logging
from dataclasses import dataclass, field

from PIL import Image

from safegrad.pipeline.hf_auth import resolve_hf_token

log = logging.getLogger(__name__)

_BATCH_TIMEOUT = 0.02
_WORKER_IDLE_TIMEOUT = 10.0


@dataclass
class _VisionRequest:
    system_prompt: str
    user_prompt: str
    image_b64: str
    temperature: float
    max_new_tokens: int
    future: asyncio.Future = field(compare=False)


class LocalVisionModel:
    """Local HuggingFace vision-language model wrapper with dynamic batching.

    Concurrent ``complete()`` calls are coalesced into a single ``pipe()``
    batch call, saturating the GPU instead of running one inference at a time.

    Parameters
    ----------
    max_batch_size:
        Maximum number of image-text pairs processed in a single forward pass.
    """

    def __init__(self, model_name: str, max_batch_size: int = 4) -> None:
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self._load_lock = asyncio.Lock()
        self._pipe = None
        self._queue: asyncio.Queue[_VisionRequest] | None = None
        self._worker_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def _ensure_loaded(self):
        if self._pipe is not None:
            return self._pipe
        async with self._load_lock:
            if self._pipe is not None:
                return self._pipe
            self._pipe = await asyncio.to_thread(self._load_sync)
            return self._pipe

    def _load_sync(self):
        import torch
        from transformers import pipeline

        log.info("Loading local vision model: %s", self.model_name)
        load_kwargs = {
            "model": self.model_name,
            "trust_remote_code": True,
        }
        token = resolve_hf_token()
        if token:
            load_kwargs["token"] = token
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
            load_kwargs["dtype"] = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            load_kwargs["model_kwargs"] = {"attn_implementation": "sdpa"}
        pipe = pipeline("image-text-to-text", **load_kwargs)

        # Disable forced thinking mode in Qwen3-VL-*-Thinking models.
        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            _THINK_PATCHES = [
                (
                    r"<|im_start|>assistant\n<think>\n",
                    r"<|im_start|>assistant\n<think>\n</think>\n\n",
                ),
                (
                    "<|im_start|>assistant\n<think>\n",
                    "<|im_start|>assistant\n<think>\n</think>\n\n",
                ),
                (
                    r"<|im_start|>assistant\n<think>",
                    r"<|im_start|>assistant\n<think></think>\n\n",
                ),
                (
                    "<|im_start|>assistant\n<think>",
                    "<|im_start|>assistant\n<think></think>\n\n",
                ),
            ]
            patched = False
            for search, replacement in _THINK_PATCHES:
                if search in tokenizer.chat_template:
                    tokenizer.chat_template = tokenizer.chat_template.replace(
                        search, replacement, 1
                    )
                    log.info("Patched tokenizer (pattern %r): thinking mode bypassed.", search[:30])
                    patched = True
                    break
            if not patched:
                log.warning(
                    "Could not patch chat_template: thinking mode still active. "
                    "Responses may contain verbose reasoning before the JSON answer."
                )

        log.info("Local vision model ready: %s", self.model_name)

        # Suppress "Both max_new_tokens and max_length" warning — the model's
        # default generation_config often has max_length=20, which conflicts with
        # our max_new_tokens setting. Setting it to None lets max_new_tokens win cleanly.
        if hasattr(pipe, "model") and hasattr(pipe.model, "generation_config"):
            pipe.model.generation_config.max_length = None

        return pipe

    @staticmethod
    def _decode_image(image_b64: str) -> Image.Image:
        raw = base64.b64decode(image_b64)
        with Image.open(io.BytesIO(raw)) as img:
            return img.convert("RGB")

    @staticmethod
    def _extract_generated_text(output) -> str:
        if isinstance(output, str):
            return output.strip()
        if isinstance(output, dict):
            if "generated_text" in output:
                return LocalVisionModel._extract_generated_text(output["generated_text"])
            if "content" in output:
                return LocalVisionModel._extract_generated_text(output["content"])
            return str(output).strip()
        if isinstance(output, list):
            if not output:
                return ""
            last = output[-1]
            if isinstance(last, dict) and last.get("role") == "assistant":
                return LocalVisionModel._extract_generated_text(last.get("content", ""))
            return LocalVisionModel._extract_generated_text(last)
        return str(output).strip()

    def _generate_batch_sync(
        self,
        pipe,
        batch: list[_VisionRequest],
    ) -> list[str]:
        """Run a single pipe() call over all requests in *batch*.

        Each request becomes one conversation (list of message dicts).  Passing
        a list of conversations to ImageTextToTextPipeline triggers its
        built-in batching path which handles variable-length padding.
        """
        conversations = []
        for req in batch:
            image = self._decode_image(req.image_b64)
            conversations.append([
                {"role": "system", "content": [{"type": "text", "text": req.system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": req.user_prompt},
                    ],
                },
            ])

        max_new_tokens = max(req.max_new_tokens for req in batch)
        temperature = batch[0].temperature

        gen_kwargs: dict = {"max_new_tokens": max_new_tokens}
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        if len(conversations) == 1:
            outputs = pipe(
                text=conversations[0],
                return_full_text=False,
                generate_kwargs=gen_kwargs,
            )
            return [self._extract_generated_text(outputs)]
        else:
            outputs = pipe(
                text=conversations,
                return_full_text=False,
                generate_kwargs=gen_kwargs,
            )
            return [self._extract_generated_text(out) for out in outputs]

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _get_queue(self) -> asyncio.Queue[_VisionRequest]:
        loop = asyncio.get_running_loop()
        if self._queue is None or self._loop is not loop:
            self._queue = asyncio.Queue()
            self._worker_task = None
            self._loop = loop
        return self._queue

    async def _ensure_worker(self) -> None:
        self._get_queue()
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._batch_worker())

    async def _batch_worker(self) -> None:
        queue = self._get_queue()
        pipe = await self._ensure_loaded()

        while True:
            try:
                first = await asyncio.wait_for(queue.get(), timeout=_WORKER_IDLE_TIMEOUT)
            except asyncio.TimeoutError:
                log.debug("Vision batch worker idle timeout, exiting.")
                return

            batch: list[_VisionRequest] = [first]

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

            log.debug("Vision: flushing batch of %d request(s).", len(batch))
            try:
                results = await asyncio.to_thread(
                    self._generate_batch_sync, pipe, batch
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
        image_b64: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> str:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        req = _VisionRequest(system_prompt, user_prompt, image_b64, temperature, max_new_tokens, fut)
        queue = self._get_queue()
        await self._ensure_worker()
        await queue.put(req)
        return await fut

    def unload(self) -> None:
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
            self._worker_task = None
        if self._pipe is None:
            return
        model = getattr(self._pipe, "model", None)
        try:
            if model is not None and hasattr(model, "to"):
                model.to("cpu")
        except Exception:
            pass
        self._pipe = None
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_LOCAL_VISION_CACHE: dict[str, LocalVisionModel] = {}


def get_local_vision_model(model_name: str, max_batch_size: int = 4) -> LocalVisionModel:
    if model_name not in _LOCAL_VISION_CACHE:
        _LOCAL_VISION_CACHE[model_name] = LocalVisionModel(model_name, max_batch_size)
    return _LOCAL_VISION_CACHE[model_name]

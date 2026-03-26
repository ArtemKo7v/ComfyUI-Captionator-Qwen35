from __future__ import annotations

import logging
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from PIL import Image

import folder_paths

try:
    import torch
    from transformers import (
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
        GenerationConfig,
        Qwen3_5ForConditionalGeneration,
    )
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    AutoProcessor = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    BitsAndBytesConfig = None  # type: ignore[assignment]
    GenerationConfig = None  # type: ignore[assignment]
    Qwen3_5ForConditionalGeneration = None  # type: ignore[assignment]

BASE_PATH = Path(folder_paths.base_path)
_DEVICE = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu") if torch else None
_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}
_NO_MODEL_SENTINEL = "No .safetensors models found"
IMAGE_FACTOR = 32
IMPROVER_STYLE_SOURCE_MODES = (
    "Details from prompt, style from image",
    "Details from image, style from prompt",
    "Merge prompt and image details and style",
)


def _model_dirs() -> Iterable[Path]:
    dirs: set[Path] = set()
    text_paths = folder_paths.folder_names_and_paths.get("text_encoders", ([], set()))[0]
    for raw in text_paths:
        dirs.add(Path(raw))

    dirs.add(Path(folder_paths.models_dir) / "llm")
    dirs.add(Path(folder_paths.models_dir) / "LLM")
    return sorted(dirs)


def _list_qwen35_models() -> Iterable[str]:
    models = []
    for model_dir in _model_dirs():
        if not model_dir.is_dir():
            continue

        for path in sorted(model_dir.rglob("*.safetensors")):
            path_text = path.as_posix().lower()
            if "qwen" not in path_text or "3" not in path_text or "5" not in path_text:
                continue
            try:
                models.append(path.relative_to(BASE_PATH).as_posix())
            except ValueError:
                models.append(path.as_posix())

    return models or [_NO_MODEL_SENTINEL]


def _resolve_model_directory(model_path: Path) -> Path:
    if model_path.is_dir():
        return model_path
    if model_path.exists():
        return model_path.parent
    return model_path.parent if model_path.parent.exists() else model_path


def _safe_bitsandbytes_config():
    if not BitsAndBytesConfig or torch is None:
        return {}
    return dict(
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    )


def _build_model_kwargs() -> Dict[str, Any]:
    kwargs = dict(
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
    )
    if _DEVICE and _DEVICE.type == "cuda":
        kwargs.update(
            dict(
                device_map="auto",
                torch_dtype=torch.float16,
                **_safe_bitsandbytes_config(),
            )
        )
    else:
        kwargs.update(device_map="cpu", torch_dtype=torch.float32)
    return kwargs


def _ensure_model(model_path: Path) -> Tuple[Any, Any, Any]:
    if (
        AutoProcessor is None
        or AutoTokenizer is None
        or Qwen3_5ForConditionalGeneration is None
        or torch is None
    ):
        raise RuntimeError("Install torch + transformers with Qwen3.5 support to use this node.")

    model_dir = _resolve_model_directory(model_path)
    key = model_dir.as_posix()
    cached = _MODEL_CACHE.get(key)
    if cached:
        return cached

    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        add_bos_token=False,
        add_eos_token=False,
    )
    model_kwargs = _build_model_kwargs()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(model_dir), **model_kwargs)
    model.eval()

    _MODEL_CACHE[key] = (processor, tokenizer, model)
    return processor, tokenizer, model


def _resize_to_limit(image: Image.Image, resize_to: int) -> Image.Image:
    width, height = image.size
    if resize_to <= 0 or max(width, height) <= resize_to:
        return image

    ratio = resize_to / max(width, height)
    width = width * ratio
    height = height * ratio
    width = ceil(width / IMAGE_FACTOR) * IMAGE_FACTOR
    height = ceil(height / IMAGE_FACTOR) * IMAGE_FACTOR
    return image.resize((int(width), int(height)), resample=Image.BICUBIC)


def _ensure_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image

    if isinstance(image, dict) and "image" in image:
        return _ensure_pil_image(image["image"])

    if torch is not None and isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        array = tensor.numpy()
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255.0).astype(np.uint8)
        img = Image.fromarray(array)
        return img.convert("RGB")

    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            array = np.clip(array, 0.0, 1.0)
            array = (array * 255.0).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(image)}")


def _build_messages(image: Image.Image | None, prompt: str) -> list[Dict[str, Any]]:
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    if prompt.strip():
        content.append({"type": "text", "text": prompt.strip()})
    return [{"role": "user", "content": content}]


def _prepare_inputs(processor: Any, image: Any | None, prompt: str, resize_to: int, think: bool) -> Dict[str, Any]:
    pil_image = None
    if image is not None:
        pil_image = _ensure_pil_image(image)
        pil_image = _resize_to_limit(pil_image, resize_to)
    messages = _build_messages(pil_image, prompt)
    template_kwargs = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=think,
    )
    try:
        inputs = processor.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        inputs = processor.apply_chat_template(messages, **template_kwargs)
    inputs.pop("token_type_ids", None)

    if _DEVICE:
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
    return inputs


def _apply_seed(seed: int) -> None:
    if torch is None or seed < 0:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _generate_text(tokenizer: Any, model: Any, inputs: Dict[str, Any], seed: int, max_new_tokens: int) -> str:
    gen_kwargs = dict(max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.95, use_cache=True, do_sample=True)
    _apply_seed(seed)
    with torch.no_grad(), torch.inference_mode():
        if GenerationConfig:
            gen_config = GenerationConfig(**gen_kwargs)
            output = model.generate(**inputs, generation_config=gen_config)
        else:
            output = model.generate(**inputs, **gen_kwargs)

    input_ids = inputs["input_ids"]
    trimmed = [sequence[input_ids.shape[1] :] for sequence in output]
    return tokenizer.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()


def _extract_caption(full_output: str, think: bool) -> str:
    if think and "</think>" in full_output:
        return full_output.split("</think>", 1)[1].strip()
    return full_output.strip()


def _build_improver_prompt(
    original_prompt: str, has_image: bool, max_new_tokens: int, style_source_mode: str
) -> str:
    original_prompt = original_prompt.strip()
    based_on_suffix = ""
    if original_prompt and has_image:
        based_on_suffix = " based on the original prompt and attached image"
    elif original_prompt:
        based_on_suffix = " based on the original prompt"
    elif has_image:
        based_on_suffix = " based on the original image"

    parts = [""]
    if original_prompt:
        parts.append(f"Original prompt: {original_prompt}\n\n")
    parts.append(f"TASK: Make a brief detailed prompt{based_on_suffix}.\n\nINSTRUCTIONS:\n")
    if original_prompt and has_image:
        if style_source_mode == IMPROVER_STYLE_SOURCE_MODES[0]:
            parts.append("Use the original prompt as the main source of subjects and content details.\n")
            parts.append("Use the attached image as the main source of visual style and stylistic details.\n")
            parts.append("If the original prompt and image conflict, keep the subjects and core details from the original prompt.\n")
        elif style_source_mode == IMPROVER_STYLE_SOURCE_MODES[1]:
            parts.append("Use the attached image as the main source of subjects and content details.\n")
            parts.append("Use the original prompt as the main source of style and stylistic cues.\n")
            parts.append("If the original prompt and image conflict, keep the subjects and core details from the attached image.\n")
        else:
            parts.append("Merge the subjects, details, and style cues from both the original prompt and the attached image.\n")
            parts.append("Combine them into one coherent prompt without repeating the same idea twice.\n")
    else:
        if original_prompt:
            parts.append("Use the specified original prompt as the source of subjects, details, and style cues.\n")
        if has_image:
            parts.append("Use the attached image as the source of subjects, details, and style cues.\n")
    parts.append(
        f"The result prompt should be in English language, be a single paragraph, and not exceed {max_new_tokens} tokens."
    )
    return "".join(parts)


class CaptionatorQwen35:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (_list_qwen35_models(),),
                "prompt": ("STRING", {"default": "Describe this image in detail.", "multiline": True}),
                "resize_to": ("INT", {"default": 0, "min": 0, "max": 4096, "step": IMAGE_FACTOR}),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 8192, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
                "think": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("caption", "full_output")
    FUNCTION = "run"
    CATEGORY = "Captionator"

    def run(self, image, model, prompt, resize_to, max_new_tokens, seed, think):
        if model == _NO_MODEL_SENTINEL:
            message = "Install a Qwen3.5 `.safetensors` with tokenizer + config in models/llm or models/text_encoders."
            return (message, message)

        model_path = (BASE_PATH / model).resolve()
        try:
            processor, tokenizer, llm = _ensure_model(model_path)
        except Exception as exc:
            logging.exception("Failed to load qwen model", exc_info=exc)
            message = f"Model load failed: {exc}"
            return (message, message)

        try:
            inputs = _prepare_inputs(processor, image, prompt, resize_to, think)
            full_output = _generate_text(tokenizer, llm, inputs, seed, max_new_tokens)
        except Exception as exc:
            logging.exception("Inference failure", exc_info=exc)
            return (f"Inference failed: {exc}", f"Inference failed: {exc}")

        caption = _extract_caption(full_output, think)
        return (caption, full_output.strip())


class CaptionImproverQwen35:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_list_qwen35_models(),),
                "prompt": ("STRING", {"default": "Enter your prompt to improve here.", "multiline": True}),
                "style_source_mode": (IMPROVER_STYLE_SOURCE_MODES,),
                "resize_to": ("INT", {"default": 512, "min": 0, "max": 4096, "step": IMAGE_FACTOR}),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 8192, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
                "think": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "full_output", "instructions_prompt")
    FUNCTION = "run"
    CATEGORY = "Captionator"

    def run(self, model, prompt, style_source_mode, resize_to, max_new_tokens, seed, think, image=None):
        if model == _NO_MODEL_SENTINEL:
            message = "Install a Qwen3.5 `.safetensors` with tokenizer + config in models/llm or models/text_encoders."
            return (message, message, message)

        original_prompt = prompt.strip()
        has_image = image is not None
        if not original_prompt and not has_image:
            message = "Provide an original prompt, an image, or both."
            return (message, message, message)

        instruction = _build_improver_prompt(original_prompt, has_image, max_new_tokens, style_source_mode)
        model_path = (BASE_PATH / model).resolve()
        try:
            processor, tokenizer, llm = _ensure_model(model_path)
        except Exception as exc:
            logging.exception("Failed to load qwen model", exc_info=exc)
            message = f"Model load failed: {exc}"
            return (message, message, instruction)

        try:
            inputs = _prepare_inputs(processor, image, instruction, resize_to, think)
            full_output = _generate_text(tokenizer, llm, inputs, seed, max_new_tokens)
        except Exception as exc:
            logging.exception("Inference failure", exc_info=exc)
            message = f"Inference failed: {exc}"
            return (message, message, instruction)

        improved_prompt = _extract_caption(full_output, think)
        return (improved_prompt, full_output.strip(), instruction)


NODE_CLASS_MAPPINGS = {
    "CaptionatorQwen35": CaptionatorQwen35,
    "CaptionImproverQwen35": CaptionImproverQwen35,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CaptionatorQwen35": "Image Captionator Qwen 3.5",
    "CaptionImproverQwen35": "Caption Improver Qwen 3.5",
}

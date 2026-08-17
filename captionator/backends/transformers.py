from __future__ import annotations

import logging
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from PIL import Image

import folder_paths

try:
    import transformers
    import torch
    from transformers import (
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
        GenerationConfig,
    )
except ImportError:  # pragma: no cover
    transformers = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    AutoProcessor = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    BitsAndBytesConfig = None  # type: ignore[assignment]
    GenerationConfig = None  # type: ignore[assignment]

from .base import CaptionBackend, GenerationRequest, GenerationResult


BASE_PATH = Path(folder_paths.base_path)
_DEVICE = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu") if torch else None
IMAGE_FACTOR = 32


@dataclass(frozen=True)
class TransformersModel:
    cache_key: str
    processor: Any | None
    tokenizer: Any
    model: Any


_MODEL_CACHE: Dict[str, TransformersModel] = {}


def _has_hf_config(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def _model_dirs() -> Iterable[Path]:
    dirs: set[Path] = set()
    text_paths = folder_paths.folder_names_and_paths.get("text_encoders", ([], set()))[0]
    for raw in text_paths:
        dirs.add(Path(raw))

    dirs.add(Path(folder_paths.models_dir) / "llm")
    dirs.add(Path(folder_paths.models_dir) / "LLM")
    return sorted(dirs)


def _find_hf_model_dir_for_path(path: Path) -> Path | None:
    start = path if path.is_dir() else path.parent
    for candidate in (start, *start.parents):
        if _has_hf_config(candidate):
            return candidate
        if candidate in _model_dirs():
            break
    return None


def _resolve_model_directory(model_path: Path) -> Path:
    model_dir = _find_hf_model_dir_for_path(model_path)
    if model_dir is not None:
        return model_dir

    if model_path.suffix == ".safetensors":
        raise RuntimeError(
            f"Selected file `{model_path.name}` is a standalone weights file. "
            "Transformers needs the full Hugging Face model directory with config.json, tokenizer files, "
            "and processor files for image input."
        )

    raise RuntimeError(f"Could not find a Hugging Face model directory with config.json for `{model_path}`.")


def _safe_bitsandbytes_config():
    if not BitsAndBytesConfig or torch is None:
        return {}
    return dict(
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    )


def _offload_folder() -> str:
    temp_dir_getter = getattr(folder_paths, "get_temp_directory", None)
    if callable(temp_dir_getter):
        offload_dir = Path(temp_dir_getter()) / "qwen35_transformers_offload"
    else:
        offload_dir = BASE_PATH / "temp" / "qwen35_transformers_offload"
    offload_dir.mkdir(parents=True, exist_ok=True)
    return str(offload_dir)


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
                offload_folder=_offload_folder(),
                torch_dtype=torch.float16,
                **_safe_bitsandbytes_config(),
            )
        )
    else:
        kwargs.update(device_map="cpu", torch_dtype=torch.float32)
    return kwargs


def _without_quantization_config(model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = dict(model_kwargs)
    kwargs.pop("quantization_config", None)
    return kwargs


def _model_loader_candidates(allow_text_fallback: bool) -> Iterable[Tuple[str, Any]]:
    if transformers is None:
        return []

    names = (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "Qwen3_5ForConditionalGeneration",
    )
    if allow_text_fallback:
        names = (*names, "AutoModelForCausalLM")

    candidates = []
    for name in names:
        loader = getattr(transformers, name, None)
        if loader is not None:
            candidates.append((name, loader))
    return candidates


def _is_bitsandbytes_params4bit_error(errors: Iterable[str]) -> bool:
    return any("Params4bit.__new__()" in error and "_is_hf_initialized" in error for error in errors)


def _try_load_with_candidates(
    model_dir: Path,
    model_kwargs: Dict[str, Any],
    allow_text_fallback: bool,
) -> Tuple[Any | None, list[str]]:
    errors = []
    for loader_name, loader in _model_loader_candidates(allow_text_fallback):
        try:
            logging.info("Loading model with %s from %s", loader_name, model_dir)
            return loader.from_pretrained(str(model_dir), **model_kwargs), errors
        except Exception as exc:
            errors.append(f"{loader_name}: {exc}")
    return None, errors


def _load_model_from_pretrained(model_dir: Path, model_kwargs: Dict[str, Any], allow_text_fallback: bool) -> Any:
    model, errors = _try_load_with_candidates(model_dir, model_kwargs, allow_text_fallback)
    if model is not None:
        return model

    if "quantization_config" in model_kwargs and _is_bitsandbytes_params4bit_error(errors):
        logging.warning(
            "4-bit bitsandbytes loading failed for %s; retrying without quantization_config. "
            "Update bitsandbytes in the ComfyUI environment if this fallback uses too much VRAM.",
            model_dir,
        )
        fallback_kwargs = _without_quantization_config(model_kwargs)
        fallback_model, fallback_errors = _try_load_with_candidates(model_dir, fallback_kwargs, allow_text_fallback)
        if fallback_model is not None:
            return fallback_model
        errors.extend(f"without quantization_config - {error}" for error in fallback_errors)

    if not errors:
        raise RuntimeError("Installed transformers version does not provide a supported Qwen/VL model loader.")

    details = "\n".join(errors)
    if allow_text_fallback:
        raise RuntimeError(f"Failed to load model with available Transformers loaders:\n{details}")

    raise RuntimeError(
        "Failed to load a multimodal Qwen3.5 model with available Transformers loaders. "
        "A processor was found, so the node did not fall back to AutoModelForCausalLM because that loader "
        "cannot consume image tensors such as pixel_values and image_grid_thw.\n"
        f"{details}"
    )


def _ensure_model(model_path: Path) -> TransformersModel:
    if AutoProcessor is None or AutoTokenizer is None or transformers is None or torch is None:
        raise RuntimeError("Install torch + transformers with Qwen/VL support to use this node.")

    model_dir = _resolve_model_directory(model_path)
    key = model_dir.as_posix()
    cached = _MODEL_CACHE.get(key)
    if cached:
        return cached

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        add_bos_token=False,
        add_eos_token=False,
    )
    try:
        processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    except ValueError as exc:
        logging.warning("Processor is not available for %s; text-only generation can still work: %s", model_dir, exc)
        processor = None

    model_kwargs = _build_model_kwargs()
    model = _load_model_from_pretrained(model_dir, model_kwargs, allow_text_fallback=processor is None)
    model.eval()

    loaded_model = TransformersModel(key, processor, tokenizer, model)
    _MODEL_CACHE[key] = loaded_model
    return loaded_model


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


def _build_text_messages(prompt: str) -> list[Dict[str, str]]:
    return [{"role": "user", "content": prompt.strip()}]


def _apply_chat_template(chat_handler: Any, messages: list[Dict[str, Any]], think: bool) -> Dict[str, Any]:
    template_kwargs = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if _supports_enable_thinking(chat_handler):
        template_kwargs["enable_thinking"] = think

    try:
        return chat_handler.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        return chat_handler.apply_chat_template(messages, **template_kwargs)


def _supports_enable_thinking(chat_handler: Any) -> bool:
    template = getattr(chat_handler, "chat_template", None)
    if template is None:
        tokenizer = getattr(chat_handler, "tokenizer", None)
        template = getattr(tokenizer, "chat_template", None)
    return isinstance(template, str) and "enable_thinking" in template


def _prepare_text_inputs(tokenizer: Any, prompt: str, think: bool) -> Dict[str, Any]:
    if hasattr(tokenizer, "apply_chat_template"):
        return _apply_chat_template(tokenizer, _build_text_messages(prompt), think)
    return tokenizer(prompt.strip(), return_tensors="pt")


def _prepare_inputs(
    processor: Any | None,
    tokenizer: Any,
    image: Any | None,
    prompt: str,
    resize_to: int,
    think: bool,
) -> Dict[str, Any]:
    pil_image = None
    if image is not None:
        if processor is None:
            raise RuntimeError("This model does not provide a processor, so image input is not supported.")
        pil_image = _ensure_pil_image(image)
        pil_image = _resize_to_limit(pil_image, resize_to)

    if pil_image is not None:
        inputs = _apply_chat_template(processor, _build_messages(pil_image, prompt), think)
    else:
        inputs = _prepare_text_inputs(tokenizer, prompt, think)
    inputs.pop("token_type_ids", None)

    if _DEVICE:
        inputs = {key: value.to(_DEVICE) for key, value in inputs.items()}
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


class TransformersBackend(CaptionBackend):
    def load(self, model_path: Path) -> TransformersModel:
        return _ensure_model(model_path)

    def generate(self, loaded_model: TransformersModel, request: GenerationRequest) -> GenerationResult:
        inputs = _prepare_inputs(
            loaded_model.processor,
            loaded_model.tokenizer,
            request.image,
            request.prompt,
            request.resize_to,
            request.think,
        )
        full_output = _generate_text(
            loaded_model.tokenizer,
            loaded_model.model,
            inputs,
            request.seed,
            request.max_new_tokens,
        )
        return GenerationResult(full_output=full_output)

    def unload(self, loaded_model: TransformersModel) -> None:
        cached = _MODEL_CACHE.get(loaded_model.cache_key)
        if cached is loaded_model:
            _MODEL_CACHE.pop(loaded_model.cache_key, None)

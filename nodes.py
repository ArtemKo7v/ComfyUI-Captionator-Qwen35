from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from PIL import Image

import folder_paths

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

BASE_PATH = Path(folder_paths.base_path)
_DEVICE = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu") if torch else None
_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}
_NO_MODEL_SENTINEL = "No .safetensors models found"


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


def _ensure_model(model_path: Path) -> Tuple[Any, Any]:
    if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
        raise RuntimeError("Install torch and transformers to run Qwen 3.5")

    model_dir = _resolve_model_directory(model_path)
    key = model_dir.as_posix()
    cached = _MODEL_CACHE.get(key)
    if cached:
        return cached

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    model_kwargs: Dict[str, Any] = dict(
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if _DEVICE and _DEVICE.type == "cuda":
        model_kwargs.update(
            dict(
                device_map="auto",
                torch_dtype=torch.float16,
            )
        )
    else:
        model_kwargs.update(device_map="cpu")

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **model_kwargs)
    model = model.eval()

    _MODEL_CACHE[key] = (tokenizer, model)
    return tokenizer, model


def _prepare_prompt(image, prompt: str) -> str:
    pil_image = _ensure_pil_image(image)
    info = f"{pil_image.width}x{pil_image.height} ({pil_image.mode})"
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        "Image metadata: "
        f"{info}\nImage base64 (PNG):\n"
        f"{image_b64}\n\nUser prompt:\n{prompt.strip()}"
    )


def _generate_text(tokenizer: Any, model: Any, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    if _DEVICE:
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

    with torch.no_grad(), torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


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
        return Image.fromarray(array)

    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            array = np.clip(array, 0.0, 1.0)
            array = (array * 255.0).astype(np.uint8)
        return Image.fromarray(array)

    raise TypeError(f"Unsupported image type: {type(image)}")


class CaptionatorQwen35:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (_list_qwen35_models(),),
                "prompt": ("STRING", {"default": "Add short caption."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "Captionator"

    def run(self, image, model, prompt):
        if model == _NO_MODEL_SENTINEL:
            return ("Place a qwen 3.5 `.safetensors` into models/llm or models/text_encoders first.",)

        model_path = (BASE_PATH / model).resolve()
        try:
            tokenizer, llm = _ensure_model(model_path)
        except Exception as exc:
            logging.exception("Failed to load qwen model", exc_info=exc)
            return (f"Model load failed: {exc}",)

        try:
            llm_prompt = _prepare_prompt(image, prompt)
            output = _generate_text(tokenizer, llm, llm_prompt)
        except Exception as exc:
            logging.exception("Inference failure", exc_info=exc)
            return (f"Inference failed: {exc}",)

        return (output.strip(),)


NODE_CLASS_MAPPINGS = {
    "CaptionatorQwen35": CaptionatorQwen35,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CaptionatorQwen35": "CaptionatorQwen35",
}

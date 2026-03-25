from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

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
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True
    )
    model = model.eval()
    if _DEVICE:
        model = model.to(_DEVICE)

    _MODEL_CACHE[key] = (tokenizer, model)
    return tokenizer, model


def _prepare_prompt(image, prompt: str) -> str:
    info = f"{image.width}x{image.height} ({image.mode})"
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
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

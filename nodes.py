from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import folder_paths

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover
    snapshot_download = None  # type: ignore[assignment]

if __package__:
    from .captionator.backends import (
        CaptionBackend,
        GenerationRequest,
        IMAGE_FACTOR,
        TransformersBackend,
    )
else:  # Allows direct loading of nodes.py in the isolated unit tests.
    from captionator.backends import (
        CaptionBackend,
        GenerationRequest,
        IMAGE_FACTOR,
        TransformersBackend,
    )


BASE_PATH = Path(folder_paths.base_path)
IMPROVER_STYLE_SOURCE_MODES = (
    "Details from prompt, style from image",
    "Details from image, style from prompt",
    "Merge prompt and image details and style",
)
DOWNLOADABLE_QWEN35_MODELS = {
    "[Download] Qwen 3.5 2B": ("Qwen/Qwen3.5-2B", "Qwen3.5-2B"),
    "[Download] Qwen 3.5 4B": ("Qwen/Qwen3.5-4B", "Qwen3.5-4B"),
    "[Download] Qwen 3.5 9B": ("Qwen/Qwen3.5-9B", "Qwen3.5-9B"),
}

_TRANSFORMERS_BACKEND = TransformersBackend()


def _model_dirs() -> Iterable[Path]:
    dirs: set[Path] = set()
    text_paths = folder_paths.folder_names_and_paths.get("text_encoders", ([], set()))[0]
    for raw in text_paths:
        dirs.add(Path(raw))

    dirs.add(Path(folder_paths.models_dir) / "llm")
    dirs.add(Path(folder_paths.models_dir) / "LLM")
    return sorted(dirs)


def _has_hf_config(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def _find_hf_model_dir_for_path(path: Path) -> Path | None:
    start = path if path.is_dir() else path.parent
    for candidate in (start, *start.parents):
        if _has_hf_config(candidate):
            return candidate
        if candidate in _model_dirs():
            break
    return None


def _display_model_path(path: Path) -> str:
    try:
        return path.relative_to(BASE_PATH).as_posix()
    except ValueError:
        return path.as_posix()


def _list_qwen35_models() -> Iterable[str]:
    models: list[str] = []
    seen: set[str] = set()
    for model_dir in _model_dirs():
        if not model_dir.is_dir():
            continue

        for path in sorted(model_dir.rglob("*.safetensors")):
            path_text = path.as_posix().lower()
            if "qwen" not in path_text or "3" not in path_text:
                continue
            display_path = _display_model_path(_find_hf_model_dir_for_path(path) or path)
            if display_path in seen:
                continue
            seen.add(display_path)
            models.append(display_path)

    return models or list(DOWNLOADABLE_QWEN35_MODELS.keys())


def _selected_model_needs_download(model_name: str) -> bool:
    return model_name in DOWNLOADABLE_QWEN35_MODELS


def _download_qwen35_model(model_name: str) -> Path:
    if snapshot_download is None:
        raise RuntimeError("Install `huggingface_hub` to download Qwen3.5 models from Hugging Face.")

    repo_id, folder_name = DOWNLOADABLE_QWEN35_MODELS[model_name]
    model_dir = Path(folder_paths.models_dir) / "llm" / folder_name
    if any(model_dir.rglob("*.safetensors")):
        return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading %s to %s", repo_id, model_dir)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return model_dir


def _resolve_selected_model_path(model_name: str) -> Path:
    if _selected_model_needs_download(model_name):
        return _download_qwen35_model(model_name)
    return (BASE_PATH / model_name).resolve()


def _select_backend(model_path: Path) -> CaptionBackend:
    return _TRANSFORMERS_BACKEND


def _load_selected_model(model_name: str):
    model_path = _resolve_selected_model_path(model_name)
    backend = _select_backend(model_path)
    return backend, backend.load(model_path)


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
        parts.append(f"ORIGINAL PROMPT:\n{original_prompt}\n\n")
    parts.append(f"TASK: Write one improved image-generation prompt{based_on_suffix}.\n\n")
    parts.append("GOAL:\n")
    parts.append(
        "Rewrite the input into a clear, vivid, compact prompt while preserving the intended meaning and important details.\n\n"
    )
    parts.append("INSTRUCTIONS:\n")
    if original_prompt and has_image:
        if style_source_mode == IMPROVER_STYLE_SOURCE_MODES[0]:
            parts.append(
                "Primary source for subjects, actions, composition, objects, attributes, and scene details: original prompt.\n"
            )
            parts.append(
                "Secondary source for style, color palette, lighting, texture, mood, and rendering look: attached image.\n"
            )
            parts.append(
                "Do not remove, replace, or reinterpret important content details from the original prompt because of the image.\n"
            )
            parts.append(
                "If the original prompt and image conflict, keep the subjects and semantic content from the original prompt.\n"
            )
        elif style_source_mode == IMPROVER_STYLE_SOURCE_MODES[1]:
            parts.append(
                "Primary source for subjects, actions, composition, objects, attributes, and scene details: attached image.\n"
            )
            parts.append(
                "Secondary source for style wording, mood, artistic direction, and stylistic cues: original prompt.\n"
            )
            parts.append(
                "Do not overwrite the main visual content from the image with conflicting content from the original prompt.\n"
            )
            parts.append(
                "If the original prompt and image conflict, keep the subjects and semantic content from the image.\n"
            )
        else:
            parts.append("Merge the subjects, scene details, and style cues from both the original prompt and the attached image.\n")
            parts.append("Keep all important non-conflicting details from both sources.\n")
            parts.append("If details overlap, consolidate them into one stronger phrasing instead of repeating them.\n")
            parts.append("If there is a direct conflict, prefer the version that creates the most coherent final prompt.\n")
    else:
        if original_prompt:
            parts.append("Use the original prompt as the only source of subjects, details, and style cues.\n")
            parts.append("Preserve all important concrete details from the original prompt.\n")
        if has_image:
            parts.append("Use the attached image as the only source of subjects, details, style, lighting, and mood.\n")
            parts.append("Describe visible content precisely without inventing unsupported details.\n")
    parts.append("Keep the final result concise but information-dense.\n")
    parts.append("Write in English.\n")
    parts.append("Return a single paragraph only.\n")
    parts.append("Return only the final prompt text with no explanation, no labels, and no bullet points.\n")
    parts.append("Keep the final prompt under 250 words.\n")
    return "".join(parts)


class CaptionatorQwen35:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_list_qwen35_models(),),
                "prompt": (
                    "STRING",
                    {
                        "default": "Write a clear and detailed description of the given image in one concise paragraph (maximum 200 words). Focus on key visual elements such as main subjects, their appearance, positions, actions, environment, lighting, colors, mood, and any notable details. Avoid speculation or assumptions beyond what is visible. Use precise, descriptive language while keeping the text compact and well-structured.",
                        "multiline": True,
                    },
                ),
                "resize_to": ("INT", {"default": 0, "min": 0, "max": 4096, "step": IMAGE_FACTOR}),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 8192, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
                "think": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("caption", "full_output")
    FUNCTION = "run"
    CATEGORY = "Captionator"

    def run(self, model, prompt, resize_to, max_new_tokens, seed, think, image=None):
        try:
            backend, loaded_model = _load_selected_model(model)
        except Exception as exc:
            logging.exception("Failed to load qwen model", exc_info=exc)
            message = f"Model load failed: {exc}"
            return (message, message)

        request = GenerationRequest(
            prompt=prompt,
            image=image,
            resize_to=resize_to,
            max_new_tokens=max_new_tokens,
            seed=seed,
            think=think,
        )
        try:
            full_output = backend.generate(loaded_model, request).full_output
        except Exception as exc:
            logging.exception("Inference failure", exc_info=exc)
            message = f"Inference failed: {exc}"
            return (message, message)

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
        original_prompt = prompt.strip()
        has_image = image is not None
        if not original_prompt and not has_image:
            message = "Provide an original prompt, an image, or both."
            return (message, message, message)

        instruction = _build_improver_prompt(original_prompt, has_image, max_new_tokens, style_source_mode)
        try:
            backend, loaded_model = _load_selected_model(model)
        except Exception as exc:
            logging.exception("Failed to load qwen model", exc_info=exc)
            message = f"Model load failed: {exc}"
            return (message, message, instruction)

        request = GenerationRequest(
            prompt=instruction,
            image=image,
            resize_to=resize_to,
            max_new_tokens=max_new_tokens,
            seed=seed,
            think=think,
        )
        try:
            full_output = backend.generate(loaded_model, request).full_output
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

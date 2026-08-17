from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_nodes_module():
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.base_path = str(PROJECT_ROOT)
    folder_paths.models_dir = str(PROJECT_ROOT / "models")
    folder_paths.folder_names_and_paths = {"text_encoders": ([], set())}

    module_name = "captionator_nodes_under_test"
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_ROOT / "nodes.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create an import specification for nodes.py")

    module = importlib.util.module_from_spec(spec)
    missing = object()
    previous_folder_paths = sys.modules.get("folder_paths", missing)
    try:
        sys.modules["folder_paths"] = folder_paths
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        transformers_backend = sys.modules["captionator.backends.transformers"]
    finally:
        if previous_folder_paths is missing:
            sys.modules.pop("folder_paths", None)
        else:
            sys.modules["folder_paths"] = previous_folder_paths
    return module, transformers_backend


nodes, transformers_backend = _load_nodes_module()


class ModelDiscoveryTests(unittest.TestCase):
    def test_lists_hugging_face_directory_once_for_sharded_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            search_root = base_path / "models" / "text_encoders"
            model_dir = search_root / "Qwen3.5-2B"
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model-00001-of-00002.safetensors").touch()
            (model_dir / "model-00002-of-00002.safetensors").touch()

            with (
                mock.patch.object(nodes, "BASE_PATH", base_path),
                mock.patch.object(nodes, "_model_dirs", return_value=[search_root]),
            ):
                models = list(nodes._list_qwen35_models())

            self.assertEqual(models, ["models/text_encoders/Qwen3.5-2B"])

    def test_lists_standalone_qwen_weights_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            search_root = base_path / "models" / "llm"
            search_root.mkdir(parents=True)
            weights = search_root / "qwen3.5-4b-fp8.safetensors"
            weights.touch()

            with (
                mock.patch.object(nodes, "BASE_PATH", base_path),
                mock.patch.object(nodes, "_model_dirs", return_value=[search_root]),
            ):
                models = list(nodes._list_qwen35_models())

            self.assertEqual(models, ["models/llm/qwen3.5-4b-fp8.safetensors"])

    def test_returns_download_actions_when_no_local_models_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(nodes, "_model_dirs", return_value=[Path(temp_dir)]):
                models = list(nodes._list_qwen35_models())

        self.assertEqual(models, list(nodes.DOWNLOADABLE_QWEN35_MODELS))

    def test_resolves_hugging_face_directory_from_nested_weight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            search_root = Path(temp_dir)
            model_dir = search_root / "Qwen3.5-9B"
            shard_dir = model_dir / "weights"
            shard_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            weights = shard_dir / "qwen3-model.safetensors"
            weights.touch()

            with mock.patch.object(transformers_backend, "_model_dirs", return_value=[search_root]):
                resolved = transformers_backend._resolve_model_directory(weights)

            self.assertEqual(resolved, model_dir)

    def test_rejects_standalone_weights_for_transformers_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            search_root = Path(temp_dir)
            weights = search_root / "qwen3.5-2b.safetensors"
            weights.touch()

            with (
                mock.patch.object(transformers_backend, "_model_dirs", return_value=[search_root]),
                self.assertRaisesRegex(RuntimeError, "standalone weights file"),
            ):
                transformers_backend._resolve_model_directory(weights)


class ImageConversionTests(unittest.TestCase):
    def test_returns_existing_pil_image_unchanged(self):
        image = Image.new("RGB", (4, 3), color=(1, 2, 3))
        self.assertIs(transformers_backend._ensure_pil_image(image), image)

    def test_converts_float_numpy_image_to_rgb(self):
        array = np.zeros((2, 3, 3), dtype=np.float32)
        array[..., 0] = 1.0

        image = transformers_backend._ensure_pil_image(array)

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (3, 2))
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))

    def test_converts_channel_first_numpy_image(self):
        array = np.zeros((3, 2, 5), dtype=np.float32)
        array[1, ...] = 1.0

        image = transformers_backend._ensure_pil_image(array)

        self.assertEqual(image.size, (5, 2))
        self.assertEqual(image.getpixel((0, 0)), (0, 255, 0))

    def test_converts_single_item_comfyui_tensor_batch(self):
        array = np.zeros((1, 2, 3, 3), dtype=np.float32)
        array[..., 2] = 1.0

        class FakeTensor:
            def __init__(self, value):
                self.value = value

            @property
            def ndim(self):
                return self.value.ndim

            @property
            def shape(self):
                return self.value.shape

            def __getitem__(self, item):
                return FakeTensor(self.value[item])

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        fake_torch = types.SimpleNamespace(Tensor=FakeTensor)
        with mock.patch.object(transformers_backend, "torch", fake_torch):
            image = transformers_backend._ensure_pil_image(FakeTensor(array))

        self.assertEqual(image.size, (3, 2))
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))

    def test_unwraps_image_dictionary(self):
        image = Image.new("RGB", (2, 2))
        self.assertIs(transformers_backend._ensure_pil_image({"image": image}), image)

    def test_rejects_unsupported_image_type(self):
        with self.assertRaisesRegex(TypeError, "Unsupported image type"):
            transformers_backend._ensure_pil_image("not-an-image")

    def test_resizes_longest_side_and_preserves_alignment(self):
        image = Image.new("RGB", (128, 64))

        resized = transformers_backend._resize_to_limit(image, 64)

        self.assertEqual(resized.size, (64, 32))

    def test_resize_zero_keeps_original_image(self):
        image = Image.new("RGB", (128, 64))
        self.assertIs(transformers_backend._resize_to_limit(image, 0), image)


class PromptPreparationTests(unittest.TestCase):
    def test_builds_multimodal_message_in_image_then_text_order(self):
        image = Image.new("RGB", (2, 2))

        messages = transformers_backend._build_messages(image, "  describe this  ")

        content = messages[0]["content"]
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(content[0], {"type": "image", "image": image})
        self.assertEqual(content[1], {"type": "text", "text": "describe this"})

    def test_builds_text_only_message(self):
        self.assertEqual(
            transformers_backend._build_text_messages("  hello  "),
            [{"role": "user", "content": "hello"}],
        )

    def test_passes_thinking_flag_when_template_supports_it(self):
        class Handler:
            chat_template = "{% if enable_thinking %}think{% endif %}"

            def apply_chat_template(self, messages, **kwargs):
                return {"messages": messages, "kwargs": kwargs}

        result = transformers_backend._apply_chat_template(Handler(), [{"role": "user"}], True)

        self.assertTrue(result["kwargs"]["enable_thinking"])
        self.assertTrue(result["kwargs"]["tokenize"])

    def test_retries_template_without_thinking_flag_on_type_error(self):
        class LegacyHandler:
            chat_template = "enable_thinking"

            def __init__(self):
                self.calls = []

            def apply_chat_template(self, messages, **kwargs):
                self.calls.append(kwargs.copy())
                if "enable_thinking" in kwargs:
                    raise TypeError("unsupported keyword")
                return {"input_ids": "ok"}

        handler = LegacyHandler()
        result = transformers_backend._apply_chat_template(handler, [{"role": "user"}], True)

        self.assertEqual(result, {"input_ids": "ok"})
        self.assertEqual(len(handler.calls), 2)
        self.assertNotIn("enable_thinking", handler.calls[-1])


class OutputAndImproverTests(unittest.TestCase):
    def test_extracts_visible_caption_after_thinking_block(self):
        output = "<think>private reasoning</think>\nVisible caption"
        self.assertEqual(nodes._extract_caption(output, True), "Visible caption")

    def test_keeps_full_output_when_thinking_is_disabled(self):
        output = "<think>reasoning</think> answer"
        self.assertEqual(nodes._extract_caption(output, False), output)

    def test_builds_prompt_only_improver_instruction(self):
        instruction = nodes._build_improver_prompt(
            "a red fox", False, 256, nodes.IMPROVER_STYLE_SOURCE_MODES[0]
        )

        self.assertIn("ORIGINAL PROMPT:\na red fox", instruction)
        self.assertIn("only source of subjects", instruction)
        self.assertIn("Return a single paragraph only", instruction)

    def test_builds_image_only_improver_instruction(self):
        instruction = nodes._build_improver_prompt(
            "", True, 256, nodes.IMPROVER_STYLE_SOURCE_MODES[0]
        )

        self.assertIn("based on the original image", instruction)
        self.assertIn("attached image as the only source", instruction)

    def test_builds_selected_prompt_and_image_priority_instruction(self):
        instruction = nodes._build_improver_prompt(
            "a castle", True, 256, nodes.IMPROVER_STYLE_SOURCE_MODES[1]
        )

        self.assertIn("Primary source for subjects", instruction)
        self.assertIn("attached image", instruction)
        self.assertIn("style wording", instruction)


class ModelLoaderTests(unittest.TestCase):
    def test_model_loader_candidates_include_text_fallback_only_when_allowed(self):
        fake_transformers = types.SimpleNamespace(
            AutoModelForImageTextToText=object(),
            AutoModelForVision2Seq=object(),
            Qwen3_5ForConditionalGeneration=object(),
            AutoModelForCausalLM=object(),
        )

        with mock.patch.object(transformers_backend, "transformers", fake_transformers):
            multimodal_names = [name for name, _ in transformers_backend._model_loader_candidates(False)]
            text_names = [name for name, _ in transformers_backend._model_loader_candidates(True)]

        self.assertNotIn("AutoModelForCausalLM", multimodal_names)
        self.assertEqual(text_names[-1], "AutoModelForCausalLM")

    def test_try_load_continues_to_next_candidate(self):
        class FailingLoader:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                raise RuntimeError("first loader failed")

        sentinel_model = object()

        class SuccessfulLoader:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                return sentinel_model

        candidates = [("FailingLoader", FailingLoader), ("SuccessfulLoader", SuccessfulLoader)]
        with mock.patch.object(transformers_backend, "_model_loader_candidates", return_value=candidates):
            model, errors = transformers_backend._try_load_with_candidates(Path("model"), {}, False)

        self.assertIs(model, sentinel_model)
        self.assertEqual(errors, ["FailingLoader: first loader failed"])

    def test_retries_bitsandbytes_initialization_error_without_quantization(self):
        quantization_config = object()
        attempts = [
            (None, ["Loader: Params4bit.__new__() got _is_hf_initialized"]),
            ("loaded-model", []),
        ]

        with mock.patch.object(transformers_backend, "_try_load_with_candidates", side_effect=attempts) as loader:
            model = transformers_backend._load_model_from_pretrained(
                Path("model"),
                {"device_map": "auto", "quantization_config": quantization_config},
                allow_text_fallback=False,
            )

        self.assertEqual(model, "loaded-model")
        second_kwargs = loader.call_args_list[1].args[1]
        self.assertEqual(second_kwargs, {"device_map": "auto"})

    def test_reports_missing_supported_transformers_loader(self):
        with (
            mock.patch.object(transformers_backend, "_try_load_with_candidates", return_value=(None, [])),
            self.assertRaisesRegex(RuntimeError, "does not provide a supported"),
        ):
            transformers_backend._load_model_from_pretrained(Path("model"), {}, allow_text_fallback=False)


class BackendContractTests(unittest.TestCase):
    def test_package_entrypoint_imports_with_relative_backend_modules(self):
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.base_path = str(PROJECT_ROOT)
        folder_paths.models_dir = str(PROJECT_ROOT / "models")
        folder_paths.folder_names_and_paths = {"text_encoders": ([], set())}

        package_name = "captionator_plugin_under_test"
        spec = importlib.util.spec_from_file_location(
            package_name,
            PROJECT_ROOT / "__init__.py",
            submodule_search_locations=[str(PROJECT_ROOT)],
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        package = importlib.util.module_from_spec(spec)

        with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}):
            sys.modules[package_name] = package
            spec.loader.exec_module(package)
            self.assertIn("CaptionatorQwen35", package.NODE_CLASS_MAPPINGS)
            self.assertIn("CaptionImproverQwen35", package.NODE_CLASS_MAPPINGS)

    def test_transformers_backend_implements_caption_backend_contract(self):
        self.assertIsInstance(nodes._TRANSFORMERS_BACKEND, nodes.CaptionBackend)

    def test_transformers_backend_unload_removes_matching_cached_handle(self):
        loaded_model = transformers_backend.TransformersModel(
            cache_key="model-key",
            processor=None,
            tokenizer=object(),
            model=object(),
        )

        with mock.patch.dict(transformers_backend._MODEL_CACHE, {"model-key": loaded_model}, clear=True):
            transformers_backend.TransformersBackend().unload(loaded_model)
            self.assertNotIn("model-key", transformers_backend._MODEL_CACHE)

    def test_captionator_node_passes_generation_request_to_backend(self):
        backend = mock.Mock()
        backend.generate.return_value = types.SimpleNamespace(
            full_output="<think>internal</think> generated caption"
        )

        with mock.patch.object(nodes, "_load_selected_model", return_value=(backend, "model-handle")):
            result = nodes.CaptionatorQwen35().run(
                model="selected-model",
                prompt="describe",
                resize_to=512,
                max_new_tokens=128,
                seed=42,
                think=True,
                image="image-value",
            )

        loaded_model, request = backend.generate.call_args.args
        self.assertEqual(loaded_model, "model-handle")
        self.assertIsInstance(request, nodes.GenerationRequest)
        self.assertEqual(request.prompt, "describe")
        self.assertEqual(request.image, "image-value")
        self.assertEqual(request.resize_to, 512)
        self.assertEqual(request.max_new_tokens, 128)
        self.assertEqual(request.seed, 42)
        self.assertTrue(request.think)
        self.assertEqual(result, ("generated caption", "<think>internal</think> generated caption"))

    def test_improver_node_passes_built_instruction_to_backend(self):
        backend = mock.Mock()
        backend.generate.return_value = types.SimpleNamespace(full_output="improved prompt")

        with mock.patch.object(nodes, "_load_selected_model", return_value=(backend, "model-handle")):
            result = nodes.CaptionImproverQwen35().run(
                model="selected-model",
                prompt="a castle",
                style_source_mode=nodes.IMPROVER_STYLE_SOURCE_MODES[0],
                resize_to=256,
                max_new_tokens=64,
                seed=7,
                think=False,
            )

        request = backend.generate.call_args.args[1]
        self.assertIn("ORIGINAL PROMPT:\na castle", request.prompt)
        self.assertEqual(request.max_new_tokens, 64)
        self.assertEqual(result[0], "improved prompt")
        self.assertEqual(result[1], "improved prompt")
        self.assertEqual(result[2], request.prompt)


if __name__ == "__main__":
    unittest.main()

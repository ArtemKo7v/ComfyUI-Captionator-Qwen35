class CaptionatorTestNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "Captionator"

    def run(self, image):
        return ("test",)


NODE_CLASS_MAPPINGS = {
    "CaptionatorTestNode": CaptionatorTestNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CaptionatorTestNode": "Captionator Test",
}

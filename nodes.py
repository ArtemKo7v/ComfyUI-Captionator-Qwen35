class CaptionatorQwen35:
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
    "CaptionatorQwen35": CaptionatorQwen35,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CaptionatorQwen35": "CaptionatorQwen35",
}

from PIL import Image
from typing import List, Optional

ALL_MODELS = ["ImageReward", "Aesthetic", "PickScore", "CLIP", "HPSv2", "HPSv2.1", "MPS"]


def uniform_sample(frames: List[Image.Image], n: int = 8) -> List[Image.Image]:
    if len(frames) <= n:
        return frames
    indices = [int(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    return [frames[i] for i in indices]


class ImageQualityEvaluator:
    def __init__(self, model_names: Optional[List[str]] = None, device: str = "cuda"):
        from diffsynth.extensions.ImageQualityMetric import download_preference_model, load_preference_model
        self.models = {}
        for name in (model_names or ALL_MODELS):
            path = download_preference_model(name)
            self.models[name] = load_preference_model(name, device=device, path=path)

    def score(self, frames: List[Image.Image], prompt: str, model_name: str) -> float:
        model = self.models[model_name]
        scores = model.score(frames, prompt)
        return sum(scores) / len(scores)

    def score_all(self, frames: List[Image.Image], prompt: str) -> dict:
        return {name: self.score(frames, prompt, name) for name in self.models}

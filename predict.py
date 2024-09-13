# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import cv2
import time
import torch
import subprocess
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor

MODEL_CACHE = "model-cache"
MODEL_NAME = "Falconsai/nsfw_image_detection"
FALCON_MODEL_URL = "https://weights.replicate.delivery/default/falconai/nsfw-image-detection.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print(f"downloading took: {time.time() - start:.2f} seconds")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        t1 = time.time()
        # Download the model weights
        if not os.path.exists(MODEL_CACHE):
            download_weights(FALCON_MODEL_URL, MODEL_CACHE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the model
        self.model = AutoModelForImageClassification.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_CACHE,
        ).to(self.device).eval()
        # Load the processor
        self.processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        t2 = time.time()
        print(f"Setup took: {t2 - t1} seconds")

    @torch.inference_mode()
    def predict(
        self,
        video: Path = Input(description="Input video"),
    ) -> str:
        """Run prediction on the video"""
        cap = cv2.VideoCapture(str(video))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # We'll analyze one frame every second
        interval = fps
        nsfw_count = 0
        frames_analyzed = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frames_analyzed % interval == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                # Process the frame
                inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                predicted_label = outputs.logits.argmax(-1).item()
                
                if self.model.config.id2label[predicted_label] == "nsfw":
                    nsfw_count += 1

            frames_analyzed += 1
        cap.release()
        # If more than 10% of analyzed frames are NSFW, classify the video as NSFW
        nsfw_ratio = nsfw_count / (frames_analyzed // interval)
        output = "nsfw" if nsfw_ratio > 0.1 else "normal"
        return output

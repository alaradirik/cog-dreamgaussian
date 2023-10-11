import os
import re
import shutil
import subprocess
from typing import List

from PIL import Image
from omegaconf import OmegaConf
from utils.image_utils import preprocess
from utils.file_utils import download_model
from cog import BasePredictor, Input, Path



CHECKPOINT_URLS = [
    ("https://storage.googleapis.com/replicate-weights/DreamGaussian/sd-2-1.tar", "/src/stable-diffusion-2-1-base"),
    ("https://storage.googleapis.com/replicate-weights/DreamGaussian/zero123-xl.tar", "/src/zero123-xl-diffusers")
]


def create_from_text(prompt):
    try:
        sanitized_prompt = re.sub('[^0-9a-zA-Z]+', '_', prompt)
        cmd1 = ["python", "optimize_raw.py", "--config", "configs/text.yaml", f"prompt={prompt}", f"save_path={sanitized_prompt}", "force_cuda_rast=True"]
        cmd2 = ["python", "optimize_refined.py", "--config", "configs/text.yaml", f"prompt={prompt}", f"save_path={sanitized_prompt}", "force_cuda_rast=True"]
        cmd3 = ["xvfb-run", "python", "-m", "kiui.render", f"logs/{sanitized_prompt}.obj", "--save_video", f"/tmp/{sanitized_prompt}.mp4"]

        subprocess.run(cmd1, check=True)
        subprocess.run(cmd2, check=True)
        subprocess.run(cmd3, check=True)
        subprocess.run(["zip", "-r", "/tmp/mesh_files.zip", "logs"], check=True)
        shutil.rmtree("logs")

        return [Path("/tmp/mesh_files.zip"), Path(f"/tmp/{sanitized_prompt}.mp4")]
    except Exception as e:
        return f"Error: {str(e)}"


def create_from_image(image, **kwargs):
    try:
        sanitized_prompt = "image"
        image.save(f"{sanitized_prompt}.png")
        input_img_path = preprocess(f"{sanitized_prompt}.png", **kwargs)
        cmd2 = ["python", "optimize_raw.py", "--config", "configs/image.yaml", f"input={sanitized_prompt}_rgba.png", f"save_path={sanitized_prompt}", "force_cuda_rast=True"]
        cmd3 = ["python", "optimize_refined.py", "--config", "configs/image.yaml", f"input={sanitized_prompt}_rgba.png", f"save_path={sanitized_prompt}", "force_cuda_rast=True"]
        cmd4 = ["xvfb-run", "python", "-m", "kiui.render", f"logs/{sanitized_prompt}.obj", "--save_video", f"/tmp/{sanitized_prompt}.mp4"]

        subprocess.run(cmd2, check=True)
        subprocess.run(cmd3, check=True)
        subprocess.run(cmd4, check=True)
        subprocess.run(["zip", "-r", "/tmp/mesh_files.zip", "logs"], check=True)

        os.remove(f"{sanitized_prompt}.png")
        os.remove(f"{sanitized_prompt}_rgba.png")
        shutil.rmtree("logs")

        return [Path("/tmp/mesh_files.zip"), Path(f"/tmp/{sanitized_prompt}.mp4")]

    except Exception as e:
        return f"Error: {str(e)}"


def process_dream_gaussian(input_type, text_input, image_input):
    if input_type == "Text":
        return create_from_text(text_input)
    else:
        return create_from_image(image_input)
    

class Predictor(BasePredictor):
    def setup(self) -> None:
        if os.path.exists("logs"):
            shutil.rmtree("logs")

        for (CKPT_URL, target_folder) in CHECKPOINT_URLS:
            if not os.path.exists(target_folder):
                print("Downloading checkpoints and config...")
                download_model(CKPT_URL, target_folder)

    def predict(
        self,
        image: Path = Input(description="Input image to convert to 3D", default=None),
        text: str = Input(description="Text prompt to generate 3D object from", default=None),
        image_size: int = Input(description="Target preprocessed input image size", default=256),
        elevation: int = Input(description="Input image elevation", ge=-90, le=90, default=0),
        num_steps: int = Input(description="Number of iterations", ge=0, default=500),
        num_refinement_steps: int = Input(description="Number of refinement iterations", ge=0, default=50),
        num_point_samples: int = Input(
            description="Number of sampled points for Gaussian Splatting", ge=500, default=5000
        )
    ) -> List[Path]:

        input_type = "Text"
        if image is not None:
            input_type = "Image"
            image = Image.open(str(image))

        output = process_dream_gaussian(input_type, text, image)
        return output

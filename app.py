import io
import logging
import math
import os
import shutil
import sys
import threading
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LPF_DIR = os.path.join(BASE_DIR, "LPF")
WEIGHTS_PATH = os.path.join(LPF_DIR, "checkpoints", "LOL", "LOL.pth")
RUNS_DIR = os.path.join(BASE_DIR, "results", "web_runs")
MAX_IMAGE_PIXELS = 4_000_000
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
OUTPUT_FILES = {"original.png", "enhanced.png"}

sys.path.insert(0, LPF_DIR)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_model = None
_device = None
_model_lock = threading.Lock()
_lpips_model = None
_lpips_lock = threading.Lock()


def _read_image(upload):
    extension = os.path.splitext(upload.filename or "")[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Choose a PNG, JPG, JPEG, BMP, or WEBP image.")

    data = upload.read()
    if not data:
        raise ValueError("The uploaded image is empty.")

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Images must be 4 megapixels or smaller.")
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc


def _load_model():
    global _model, _device

    if _model is None:
        if not os.path.isfile(WEIGHTS_PATH):
            raise RuntimeError("The LPF checkpoint is missing: " + WEIGHTS_PATH)

        from model.LPF import LPF

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LPF()
        checkpoint = torch.load(WEIGHTS_PATH, map_location=_device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {
                key[7:] if key.startswith("module.") else key: value
                for key, value in state_dict.items()
            }
        model.load_state_dict(state_dict)
        model.to(_device)
        model.eval()
        _model = model
        logger.info("Loaded LPF checkpoint on %s", _device)

    return _model, _device


def _run_lpf(image, job_dir):
    model, device = _load_model()
    original = np.asarray(image, dtype=np.uint8)
    height, width = original.shape[:2]
    tensor = torch.from_numpy(original.copy()).float().div_(255)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    pad_h = (-height) % 8
    pad_w = (-width) % 8
    if pad_h or pad_w:
        padding_mode = "reflect" if height > pad_h and width > pad_w else "replicate"
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode=padding_mode)

    with torch.inference_mode():
        enhanced = model(tensor).clamp(0, 1)[:, :, :height, :width]

    enhanced_rgb = (
        enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
    ).round().astype(np.uint8)

    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(job_dir, exist_ok=False)
    Image.fromarray(original).save(os.path.join(job_dir, "original.png"))
    Image.fromarray(enhanced_rgb).save(os.path.join(job_dir, "enhanced.png"))


def _image_statistics(image):
    rgb = np.asarray(image, dtype=np.float32) / 255
    luminance = (
        0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    )
    return {
        "brightness": round(float(luminance.mean() * 100), 2),
        "contrast": round(float(luminance.std() * 100), 2),
    }


def _reference_metrics(enhanced_image, reference_image):
    global _lpips_model

    reference = reference_image.resize(
        enhanced_image.size, Image.Resampling.LANCZOS
    )
    enhanced_array = np.asarray(enhanced_image, dtype=np.float32) / 255
    reference_array = np.asarray(reference, dtype=np.float32) / 255

    mse = float(np.mean((enhanced_array - reference_array) ** 2))
    psnr = None if mse == 0 else 10 * math.log10(1 / mse)

    enhanced_tensor = (
        torch.from_numpy(enhanced_array).permute(2, 0, 1).unsqueeze(0).double()
    )
    reference_tensor = (
        torch.from_numpy(reference_array).permute(2, 0, 1).unsqueeze(0).double()
    )
    window_size = min(7, enhanced_image.width, enhanced_image.height)
    if window_size % 2 == 0:
        window_size -= 1
    # Match the project's evaluation.py: a 7x7 uniform window with sample covariance.
    window = torch.full((3, 1, window_size, window_size), 1 / window_size**2, dtype=torch.float64)
    mean_enhanced = F.conv2d(enhanced_tensor, window, groups=3)
    mean_reference = F.conv2d(reference_tensor, window, groups=3)
    mean_enhanced_sq = mean_enhanced.square()
    mean_reference_sq = mean_reference.square()
    covariance_mean = mean_enhanced * mean_reference
    variance_enhanced = (
        F.conv2d(enhanced_tensor.square(), window, groups=3)
        - mean_enhanced_sq
    )
    variance_reference = (
        F.conv2d(reference_tensor.square(), window, groups=3)
        - mean_reference_sq
    )
    covariance = (
        F.conv2d(enhanced_tensor * reference_tensor, window, groups=3)
        - covariance_mean
    )
    covariance_correction = window_size**2 / (window_size**2 - 1) if window_size > 1 else 1
    variance_enhanced = variance_enhanced * covariance_correction
    variance_reference = variance_reference * covariance_correction
    covariance = covariance * covariance_correction
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * covariance_mean + c1) * (2 * covariance + c2)) / (
        (mean_enhanced_sq + mean_reference_sq + c1)
        * (variance_enhanced + variance_reference + c2)
    )

    metrics = {
        "psnr": None if psnr is None else round(psnr, 4),
        "ssim": round(float(ssim_map.mean()), 4),
        "lpips": None,
        "lpips_note": None,
    }

    try:
        import lpips

        if _lpips_model is None:
            with _lpips_lock:
                if _lpips_model is None:
                    _lpips_model = lpips.LPIPS(net="alex").eval()
        enhanced_lpips = enhanced_tensor.float().mul(2).sub(1)
        reference_lpips = reference_tensor.float().mul(2).sub(1)
        with torch.inference_mode():
            metrics["lpips"] = round(
                float(_lpips_model(enhanced_lpips, reference_lpips).item()), 4
            )
    except ImportError:
        metrics["lpips_note"] = "Install the optional lpips package to calculate LPIPS."
    except Exception as exc:
        logger.exception("Could not calculate LPIPS")
        metrics["lpips_note"] = "LPIPS could not load its model weights: " + str(exc)

    return metrics


def _remove_expired_runs():
    if not os.path.isdir(RUNS_DIR):
        return
    expiration_time = time.time() - 24 * 60 * 60
    for entry in os.scandir(RUNS_DIR):
        if entry.is_dir(follow_symlinks=False) and entry.stat().st_mtime < expiration_time:
            shutil.rmtree(entry.path)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/enhance")
def enhance():
    try:
        upload = request.files.get("image")
        if upload is None:
            return jsonify(error="Select a low-light image to continue."), 400

        image = _read_image(upload)
        reference_upload = request.files.get("reference")
        reference = _read_image(reference_upload) if reference_upload and reference_upload.filename else None

        _remove_expired_runs()
        job_id = uuid.uuid4().hex
        job_dir = os.path.join(RUNS_DIR, job_id)

        with _model_lock:
            _run_lpf(image, job_dir)

        enhanced_image = Image.open(os.path.join(job_dir, "enhanced.png")).convert("RGB")
        result = {
            "job_id": job_id,
            "device": str(_device),
            "images": {
                name[:-4]: f"/outputs/{job_id}/{name}"
                for name in sorted(OUTPUT_FILES)
                if os.path.isfile(os.path.join(job_dir, name))
            },
            "statistics": {
                "original": _image_statistics(image),
                "enhanced": _image_statistics(enhanced_image),
            },
            "reference_metrics": (
                _reference_metrics(enhanced_image, reference) if reference else None
            ),
        }
        return jsonify(result)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except RequestEntityTooLarge:
        return jsonify(error="Each upload must be 25 MB or smaller."), 413
    except Exception:
        logger.exception("Image enhancement failed")
        return jsonify(error="Enhancement failed. Check the server terminal for details."), 500


@app.get("/outputs/<job_id>/<filename>")
def output(job_id, filename):
    if not all(character in "0123456789abcdef" for character in job_id):
        return jsonify(error="Result not found."), 404
    if filename not in OUTPUT_FILES:
        return jsonify(error="Result not found."), 404
    return send_from_directory(os.path.join(RUNS_DIR, job_id), filename)


if __name__ == "__main__":
    os.makedirs(RUNS_DIR, exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=False)

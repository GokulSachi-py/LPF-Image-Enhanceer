# Low-Light Image Enhancement

This repository contains the local LPF image-enhancement website, LPF and RetinexFormer source code and pretrained checkpoints, the LOL dataset, and the scripts and results used to compare both models.

## Run the website

```powershell
git clone https://github.com/GokulSachi-py/LPF-Image-Enhanceer.git
cd LPF-Image-Enhanceer
python -m pip install -r requirements-web.txt
python app.py
```

Open <http://127.0.0.1:5000> and keep the terminal running while using the app. It uses the included LPF code and checkpoint in `LPF/`.

## Project contents

- `app.py`, `templates/`, and `static/`: local image enhancement website.
- `LPF/`: LPF model implementation and pretrained checkpoints.
- `RetinexFormer/`: RetinexFormer implementation and pretrained checkpoints.
- `dataset/`: included LOL-v1 training and evaluation image pairs.
- `compare.py`, `enhance.py`, and `real_commands.txt`: model comparison, inference, and experiment commands.
- `results/`: saved enhanced images, visual comparisons, and evaluation metrics.

The LPF and RetinexFormer source trees retain their upstream notices and links. The LOL dataset is included for this private project; do not make the repository public or redistribute dataset/model files unless you have permission to do so.

## Metrics

PSNR, SSIM, and LPIPS require a matching well-lit reference image. Without a reference, the website shows brightness and contrast as descriptive measurements, not quality scores. See [WEB_APP.md](WEB_APP.md) for details.

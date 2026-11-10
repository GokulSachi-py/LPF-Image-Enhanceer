# LPF Image Enhancer

A small web app for uploading a low-light image, enhancing it with the pretrained LPF model, comparing the original and enhanced images, and viewing image measurements.

## Get started

Clone this repository and its LPF model dependency:

```powershell
git clone --recurse-submodules https://github.com/GokulSachi-py/LPF-Image-Enhanceer.git
cd LPF-Image-Enhanceer
python -m pip install -r requirements-web.txt
python app.py
```

Open <http://127.0.0.1:5000> and keep the terminal running while you use the website. If you already cloned without its submodule, run:

```powershell
git submodule update --init --recursive
```

The LPF model code and pretrained LOL-v1 checkpoint are provided by the [LPF repository](https://github.com/DeniJsonC/LPF), included as a Git submodule. The web app code itself lives in this repository. Large datasets, generated results, and the separate RetinexFormer repository are not part of this website repo.

## Metrics

PSNR, SSIM, and LPIPS are computed only when you provide a matching well-lit reference image. Without a reference, the app displays brightness and contrast as descriptive image measurements rather than quality scores.

See [WEB_APP.md](WEB_APP.md) for additional details.

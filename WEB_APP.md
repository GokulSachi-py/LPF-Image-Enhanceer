# LPF image enhancement website

From the project folder, install the web dependencies and start the local server:

```powershell
python -m pip install -r requirements-web.txt
python app.py
```

Keep the terminal open while using the site, then open <http://127.0.0.1:5000>. If the server is stopped, the browser cannot submit images. The app uses the existing LPF LOL-v1 checkpoint at `LPF/checkpoints/LOL/LOL.pth`.

The page shows the input and LPF-enhanced output, with a download link for the result. PNG, JPG, JPEG, BMP, and WEBP uploads are supported (up to 25 MB and 4 megapixels).

PSNR, SSIM, and LPIPS require an optional matching ground-truth/reference image. PSNR and SSIM are computed locally. LPIPS uses the installed `lpips` package and AlexNet weights; the weights download automatically the first time the metric is used. Without a reference, the page shows mean luminance and contrast as descriptive image measurements, not quality scores.

The server binds to `127.0.0.1` and is intended for local use.

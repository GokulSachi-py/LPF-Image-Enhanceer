"""
Single Image & Directory Image Enhancement Script
Supports: LPF, RetinexFormer, or Both models side-by-side
=========================================================
Usage Examples:
  # 1. Enhance a single image using both models (saves individual + side-by-side comparison):
  python enhance.py --image path/to/image.jpg

  # 2. Enhance using only LPF:
  python enhance.py --image path/to/image.jpg --model lpf

  # 3. Enhance using only RetinexFormer:
  python enhance.py --image path/to/image.jpg --model retinexformer

  # 4. Enhance all images in a folder:
  python enhance.py --input_dir path/to/folder --model both

  # 5. Specify custom output folder:
  python enhance.py --image path/to/image.jpg --output results/my_demo
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Add model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LPF_DIR = os.path.join(BASE_DIR, 'LPF')
RF_DIR = os.path.join(BASE_DIR, 'RetinexFormer')
sys.path.insert(0, LPF_DIR)
sys.path.insert(0, RF_DIR)

DEFAULT_LPF_WEIGHTS = os.path.join(LPF_DIR, 'checkpoints', 'LOL', 'LOL.pth')
DEFAULT_RF_WEIGHTS = os.path.join(RF_DIR, 'pretrained_weights', 'LOL_v1.pth')
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, 'results', 'demo_enhanced')


# ==========================================
# Model Loading Helpers
# ==========================================
def load_lpf_model(weights_path, device):
    """Load LPF model and weights."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"LPF weights not found at: {weights_path}")
    
    from model.LPF import LPF
    model = LPF()
    checkpoint = torch.load(weights_path, map_location=device)
    
    state_dict = checkpoint.get('state_dict', checkpoint)
    if any(k.startswith('module.') for k in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model


def load_retinexformer_model(weights_path, device):
    """Load RetinexFormer model and weights."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"RetinexFormer weights not found at: {weights_path}")
    
    from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
    model = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
    checkpoint = torch.load(weights_path, map_location=device)
    
    params = checkpoint.get('params', checkpoint)
    # Handle module prefix if needed
    try:
        model.load_state_dict(params)
    except Exception:
        from collections import OrderedDict
        new_params = OrderedDict()
        for k, v in params.items():
            name = k[7:] if k.startswith('module.') else k
            new_params[name] = v
        model.load_state_dict(new_params)
        
    model.to(device)
    model.eval()
    return model


# ==========================================
# Inference Functions
# ==========================================
def enhance_lpf(model, img_bgr, device):
    """Enhance a BGR image using LPF. Returns enhanced BGR image."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    h, w = img_tensor.shape[2], img_tensor.shape[3]
    mul = 8
    pad_h = (mul - h % mul) if h % mul != 0 else 0
    pad_w = (mul - w % mul) if w % mul != 0 else 0
    input_padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), 'reflect')
    
    with torch.no_grad():
        output = model(input_padded)
        output = output[:, :, :h, :w]
        output = torch.clamp(output, 0.0, 1.0)
    
    out_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_uint8 = (out_np * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)


def enhance_retinexformer(model, img_bgr, device):
    """Enhance a BGR image using RetinexFormer. Returns enhanced BGR image."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    h, w = img_tensor.shape[2], img_tensor.shape[3]
    factor = 8
    pad_h = (factor - h % factor) if h % factor != 0 else 0
    pad_w = (factor - w % factor) if w % factor != 0 else 0
    input_padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), 'reflect')
    
    with torch.no_grad():
        output = model(input_padded)
        output = output[:, :, :h, :w]
        output = torch.clamp(output, 0.0, 1.0)
    
    out_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_uint8 = (out_np * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)


def make_comparison_image(panels, labels):
    """Create a side-by-side labeled comparison image."""
    h, w = panels[0].shape[:2]
    # Resize all panels to same height and width
    resized_panels = [cv2.resize(p, (w, h)) for p in panels]
    
    label_h = 36
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    
    labeled_panels = []
    for img, label in zip(resized_panels, labels):
        bar = np.zeros((label_h, w, 3), dtype=np.uint8)
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = max((w - text_size[0]) // 2, 5)
        text_y = (label_h + text_size[1]) // 2 - 2
        cv2.putText(bar, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        labeled_panels.append(np.vstack([bar, img]))
    
    return np.hstack(labeled_panels)


# ==========================================
# Main CLI Function
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Enhance low-light images using LPF and/or RetinexFormer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python enhance.py --image dataset/LOLdataset/eval15/low/1.png
  python enhance.py --image my_photo.jpg --model lpf
  python enhance.py --image my_photo.jpg --model retinexformer
  python enhance.py --input_dir my_folder/ --output results/test_out
"""
    )
    parser.add_argument('--image', '-i', type=str, default=None,
                        help="Path to a single image file to enhance")
    parser.add_argument('--input_dir', '-d', type=str, default=None,
                        help="Path to a folder of images to enhance")
    parser.add_argument('--model', '-m', type=str, default='both',
                        choices=['both', 'lpf', 'retinexformer', 'rf'],
                        help="Model to use: 'both' (default), 'lpf', or 'retinexformer'")
    parser.add_argument('--output', '-o', type=str, default=DEFAULT_OUT_DIR,
                        help=f"Output directory to save enhanced images (default: {DEFAULT_OUT_DIR})")
    parser.add_argument('--lpf_weights', type=str, default=DEFAULT_LPF_WEIGHTS,
                        help="Path to LPF checkpoint (.pth)")
    parser.add_argument('--rf_weights', type=str, default=DEFAULT_RF_WEIGHTS,
                        help="Path to RetinexFormer checkpoint (.pth)")
    parser.add_argument('--cpu', action='store_true',
                        help="Force CPU execution even if CUDA is available")

    args = parser.parse_args()

    # If neither --image nor --input_dir is provided, provide helpful guidance
    if not args.image and not args.input_dir:
        sample_img = os.path.join(BASE_DIR, 'dataset', 'LOLdataset', 'eval15', 'low', '1.png')
        print("=" * 70)
        print("  Quick Image Enhancement Tool (LPF & RetinexFormer)")
        print("=" * 70)
        print("\nNo input image specified!")
        print("\nTo test with a single image, run:")
        print(f"  python enhance.py --image \"{sample_img}\"")
        print("\nTo test any custom image, run:")
        print("  python enhance.py --image \"path/to/any_image.jpg\"")
        print("\nOptions:")
        print("  --model both          (Enhances with both and creates a comparison)")
        print("  --model lpf           (Enhance with LPF only)")
        print("  --model retinexformer (Enhance with RetinexFormer only)")
        print("  --output <dir>        (Custom output directory)")
        print("=" * 70)
        return

    # Determine files to process
    image_paths = []
    if args.image:
        if not os.path.isfile(args.image):
            print(f"[Error] Specified image file not found: {args.image}")
            sys.exit(1)
        image_paths.append(args.image)
    elif args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"[Error] Specified directory not found: {args.input_dir}")
            sys.exit(1)
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.PNG', '.JPG', '.JPEG', '.BMP')
        for fname in sorted(os.listdir(args.input_dir)):
            if fname.endswith(valid_exts):
                image_paths.append(os.path.join(args.input_dir, fname))
        if not image_paths:
            print(f"[Error] No supported image files (.png, .jpg, .bmp) found in {args.input_dir}")
            sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print("=" * 70)
    print("  Low-Light Image Enhancement")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output directory: {os.path.abspath(args.output)}")
    print(f"Number of images to process: {len(image_paths)}")
    print(f"Selected model: {args.model}")
    print("=" * 70)

    # Load required models
    model_choice = 'rf' if args.model == 'retinexformer' else args.model
    lpf_net = None
    rf_net = None

    if model_choice in ['both', 'lpf']:
        print(f"Loading LPF model from: {args.lpf_weights} ...")
        lpf_net = load_lpf_model(args.lpf_weights, device)
        print("  -> LPF loaded successfully.")

    if model_choice in ['both', 'rf']:
        print(f"Loading RetinexFormer model from: {args.rf_weights} ...")
        rf_net = load_retinexformer_model(args.rf_weights, device)
        print("  -> RetinexFormer loaded successfully.")

    print("\nProcessing images...\n")

    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        base_name, ext = os.path.splitext(filename)
        print(f"[{idx}/{len(image_paths)}] Processing: {filename} ({img_path})")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  [Warning] Could not read image: {img_path}. Skipping.")
            continue

        orig_h, orig_w = img_bgr.shape[:2]
        print(f"  Input resolution: {orig_w}x{orig_h}")

        panels = [img_bgr]
        labels = ['Original Low-Light']

        # Run LPF
        if lpf_net is not None:
            lpf_enhanced = enhance_lpf(lpf_net, img_bgr, device)
            lpf_save_path = os.path.join(args.output, f"{base_name}_lpf.png")
            cv2.imwrite(lpf_save_path, lpf_enhanced)
            print(f"  -> LPF enhanced saved: {lpf_save_path}")
            panels.append(lpf_enhanced)
            labels.append('LPF Enhanced')

        # Run RetinexFormer
        if rf_net is not None:
            rf_enhanced = enhance_retinexformer(rf_net, img_bgr, device)
            rf_save_path = os.path.join(args.output, f"{base_name}_retinexformer.png")
            cv2.imwrite(rf_save_path, rf_enhanced)
            print(f"  -> RetinexFormer enhanced saved: {rf_save_path}")
            panels.append(rf_enhanced)
            labels.append('RetinexFormer Enhanced')

        # If both models ran, also produce a side-by-side comparison image!
        if model_choice == 'both' and len(panels) == 3:
            comp_img = make_comparison_image(panels, labels)
            comp_save_path = os.path.join(args.output, f"{base_name}_comparison.png")
            cv2.imwrite(comp_save_path, comp_img)
            print(f"  -> Side-by-side comparison saved: {comp_save_path}")

    print("\n" + "=" * 70)
    print(f"All processing complete! Check the outputs in:\n  {os.path.abspath(args.output)}")
    print("=" * 70)


if __name__ == '__main__':
    main()

"""
Full Comparison Script: LPF vs RetinexFormer on LOL-v1
======================================================
Generates:
  1. Per-image metric table (PSNR, SSIM, LPIPS)
  2. Average metric comparison table
  3. Parameter count & FLOPs comparison
  4. Visual comparison grid images
"""
import os
import sys
import numpy as np
import cv2
import torch
import argparse
from collections import OrderedDict

# Add LPF to path for model import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LPF'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'RetinexFormer'))

from natsort import natsorted
from glob import glob
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
import lpips

# ========== Configuration ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GT_DIR = os.path.join(BASE_DIR, 'dataset', 'LOLdataset', 'eval15', 'high')
LOW_DIR = os.path.join(BASE_DIR, 'dataset', 'LOLdataset', 'eval15', 'low')
LPF_DIR = os.path.join(BASE_DIR, 'results', 'LPF', 'enhanced')
RF_DIR = os.path.join(BASE_DIR, 'results', 'RetinexFormer', 'enhanced')
COMP_DIR = os.path.join(BASE_DIR, 'results', 'comparison')
os.makedirs(COMP_DIR, exist_ok=True)

# ========== Metric Helpers ==========
def imread_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_lpips_tensor(img):
    """Convert uint8 HWC image to NCHW tensor in [-1, 1]."""
    img = img.astype(np.float32) / 127.5 - 1.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)

def compute_metrics(enhanced_path, gt_path, lpips_model):
    """Compute PSNR, SSIM, LPIPS between enhanced image and ground truth."""
    enhanced = imread_rgb(enhanced_path)
    gt = imread_rgb(gt_path)
    
    # Ensure same size
    if enhanced.shape != gt.shape:
        enhanced = cv2.resize(enhanced, (gt.shape[1], gt.shape[0]))
    
    p = psnr_fn(gt, enhanced)
    s = ssim_fn(gt, enhanced, channel_axis=2)
    
    t_enh = to_lpips_tensor(enhanced)
    t_gt = to_lpips_tensor(gt)
    with torch.no_grad():
        l = lpips_model(t_enh, t_gt).item()
    
    return p, s, l

# ========== Parameter & FLOPs Counting ==========
def count_lpf_params_flops():
    """Count LPF parameters and FLOPs."""
    from model.LPF import LPF
    model = LPF()
    total_params = sum(p.numel() for p in model.parameters())
    
    # FLOPs using thop
    try:
        from thop import profile
        dummy_input = torch.randn(1, 3, 400, 600)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    except Exception as e:
        print(f"Warning: Could not compute LPF FLOPs: {e}")
        flops = None
    
    return total_params, flops

def count_retinexformer_params_flops():
    """Count RetinexFormer parameters and FLOPs."""
    try:
        from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
        model = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1,2,2])
        total_params = sum(p.numel() for p in model.parameters())
        
        try:
            from thop import profile
            dummy_input = torch.randn(1, 3, 400, 600)
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        except Exception as e:
            print(f"Warning: Could not compute RetinexFormer FLOPs: {e}")
            flops = None
        
        return total_params, flops
    except Exception as e:
        print(f"Warning: Could not load RetinexFormer model: {e}")
        return None, None

# ========== Visual Comparison ==========
def create_comparison_image(low_path, lpf_path, rf_path, gt_path, save_path):
    """Create a side-by-side comparison image: Low | LPF | RetinexFormer | GT."""
    low = cv2.imread(low_path)
    lpf = cv2.imread(lpf_path)
    rf = cv2.imread(rf_path)
    gt = cv2.imread(gt_path)
    
    if low is None or lpf is None or rf is None or gt is None:
        print(f"Warning: Skipping comparison, could not read all images for {os.path.basename(save_path)}")
        return
    
    # Resize all to same dimensions (use GT as reference)
    h, w = gt.shape[:2]
    low = cv2.resize(low, (w, h))
    lpf = cv2.resize(lpf, (w, h))
    rf = cv2.resize(rf, (w, h))
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (255, 255, 255)
    bg_color = (0, 0, 0)
    
    label_h = 35
    imgs = []
    labels = ['Low-Light Input', 'LPF Enhanced', 'RetinexFormer Enhanced', 'Ground Truth']
    for img, label in zip([low, lpf, rf, gt], labels):
        # Create label bar
        bar = np.zeros((label_h, w, 3), dtype=np.uint8)
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(bar, label, (text_x, label_h - 10), font, font_scale, color, thickness)
        imgs.append(np.vstack([bar, img]))
    
    # Concatenate horizontally
    comparison = np.hstack(imgs)
    cv2.imwrite(save_path, comparison)

# ========== Main ==========
def main():
    print("=" * 70)
    print("  LPF vs RetinexFormer - LOL-v1 Comparison")
    print("=" * 70)
    
    # Gather image filenames
    gt_files = natsorted(glob(os.path.join(GT_DIR, '*.png')))
    if not gt_files:
        print(f"ERROR: No ground-truth images found in {GT_DIR}")
        return
    
    filenames = [os.path.basename(f) for f in gt_files]
    print(f"\nFound {len(filenames)} test images: {[os.path.splitext(f)[0] for f in filenames]}")
    
    # Initialize LPIPS
    print("\nInitializing LPIPS model...")
    lpips_model = lpips.LPIPS(net='alex')
    lpips_model.eval()
    
    # Per-image metrics
    lpf_metrics = []
    rf_metrics = []
    
    print("\n--- Per-Image Metrics ---")
    header = f"{'Image':<10} | {'LPF PSNR':>9} {'LPF SSIM':>9} {'LPF LPIPS':>10} | {'RF PSNR':>9} {'RF SSIM':>9} {'RF LPIPS':>10}"
    print(header)
    print("-" * len(header))
    
    for fn in filenames:
        gt_path = os.path.join(GT_DIR, fn)
        lpf_path = os.path.join(LPF_DIR, fn)
        rf_path = os.path.join(RF_DIR, fn)
        
        name = os.path.splitext(fn)[0]
        
        if not os.path.exists(lpf_path):
            print(f"  WARNING: LPF enhanced image not found: {lpf_path}")
            continue
        if not os.path.exists(rf_path):
            print(f"  WARNING: RetinexFormer enhanced image not found: {rf_path}")
            continue
        
        lp, ls, ll = compute_metrics(lpf_path, gt_path, lpips_model)
        rp, rs, rl = compute_metrics(rf_path, gt_path, lpips_model)
        
        lpf_metrics.append({'name': name, 'psnr': lp, 'ssim': ls, 'lpips': ll})
        rf_metrics.append({'name': name, 'psnr': rp, 'ssim': rs, 'lpips': rl})
        
        print(f"{name:<10} | {lp:9.2f} {ls:9.4f} {ll:10.4f} | {rp:9.2f} {rs:9.4f} {rl:10.4f}")
    
    # Average metrics
    if lpf_metrics and rf_metrics:
        lpf_avg_psnr = np.mean([m['psnr'] for m in lpf_metrics])
        lpf_avg_ssim = np.mean([m['ssim'] for m in lpf_metrics])
        lpf_avg_lpips = np.mean([m['lpips'] for m in lpf_metrics])
        
        rf_avg_psnr = np.mean([m['psnr'] for m in rf_metrics])
        rf_avg_ssim = np.mean([m['ssim'] for m in rf_metrics])
        rf_avg_lpips = np.mean([m['lpips'] for m in rf_metrics])
        
        print("-" * len(header))
        print(f"{'Average':<10} | {lpf_avg_psnr:9.2f} {lpf_avg_ssim:9.4f} {lpf_avg_lpips:10.4f} | {rf_avg_psnr:9.2f} {rf_avg_ssim:9.4f} {rf_avg_lpips:10.4f}")
    
    # Parameter count & FLOPs
    print("\n\n--- Model Complexity ---")
    lpf_params, lpf_flops = count_lpf_params_flops()
    rf_params, rf_flops = count_retinexformer_params_flops()
    
    print(f"{'Model':<20} {'Parameters':>15} {'FLOPs (400x600)':>20}")
    print("-" * 60)
    lpf_p_str = f"{lpf_params/1e6:.4f}M" if lpf_params else "N/A"
    lpf_f_str = f"{lpf_flops/1e9:.2f}G" if lpf_flops else "N/A"
    rf_p_str = f"{rf_params/1e6:.4f}M" if rf_params else "N/A"
    rf_f_str = f"{rf_flops/1e9:.2f}G" if rf_flops else "N/A"
    
    print(f"{'LPF':<20} {lpf_p_str:>15} {lpf_f_str:>20}")
    print(f"{'RetinexFormer':<20} {rf_p_str:>15} {rf_f_str:>20}")
    
    # Final Comparison Table
    print("\n\n" + "=" * 70)
    print("  FINAL COMPARISON TABLE")
    print("=" * 70)
    print(f"| {'Model':<20} | {'Params':>10} | {'FLOPs':>10} | {'PSNR':>8} | {'SSIM':>8} | {'LPIPS':>8} |")
    print(f"|{'-'*22}|{'-'*12}|{'-'*12}|{'-'*10}|{'-'*10}|{'-'*10}|")
    print(f"| {'LPF':<20} | {lpf_p_str:>10} | {lpf_f_str:>10} | {lpf_avg_psnr:8.2f} | {lpf_avg_ssim:8.4f} | {lpf_avg_lpips:8.4f} |")
    print(f"| {'RetinexFormer':<20} | {rf_p_str:>10} | {rf_f_str:>10} | {rf_avg_psnr:8.2f} | {rf_avg_ssim:8.4f} | {rf_avg_lpips:8.4f} |")
    print("=" * 70)
    
    # Visual comparisons
    print(f"\nGenerating visual comparisons in {COMP_DIR}...")
    for fn in filenames:
        name = os.path.splitext(fn)[0]
        low_path = os.path.join(LOW_DIR, fn)
        lpf_path = os.path.join(LPF_DIR, fn)
        rf_path = os.path.join(RF_DIR, fn)
        gt_path = os.path.join(GT_DIR, fn)
        save_path = os.path.join(COMP_DIR, f'comparison_{name}.png')
        
        if all(os.path.exists(p) for p in [low_path, lpf_path, rf_path, gt_path]):
            create_comparison_image(low_path, lpf_path, rf_path, gt_path, save_path)
            print(f"  Saved: comparison_{name}.png")
    
    # Save results to CSV
    csv_path = os.path.join(BASE_DIR, 'results', 'metrics_comparison.csv')
    with open(csv_path, 'w') as f:
        f.write("Image,LPF_PSNR,LPF_SSIM,LPF_LPIPS,RetinexFormer_PSNR,RetinexFormer_SSIM,RetinexFormer_LPIPS\n")
        for lm, rm in zip(lpf_metrics, rf_metrics):
            f.write(f"{lm['name']},{lm['psnr']:.4f},{lm['ssim']:.4f},{lm['lpips']:.4f},{rm['psnr']:.4f},{rm['ssim']:.4f},{rm['lpips']:.4f}\n")
        f.write(f"Average,{lpf_avg_psnr:.4f},{lpf_avg_ssim:.4f},{lpf_avg_lpips:.4f},{rf_avg_psnr:.4f},{rf_avg_ssim:.4f},{rf_avg_lpips:.4f}\n")
    print(f"\nMetrics saved to: {csv_path}")
    
    print("\n[DONE] Comparison complete!")

if __name__ == '__main__':
    main()

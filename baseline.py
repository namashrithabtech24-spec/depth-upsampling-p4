import os, glob, time
import numpy as np
import cv2
from utils import compute_metrics, boundary_f1, edge_rmse
from hybrid import adaptive_fusion, guided_filter_upsample

SCALES = [2, 4, 8]

def bicubic_upsample(lr, size_hw):
    return cv2.resize(lr, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_CUBIC)

def downsample_depth(hr, scale):
    h, w = hr.shape
    return cv2.resize(hr, (w // scale, h // scale), interpolation=cv2.INTER_AREA)

# Simulate LiDAR/ToF degradation: random dropout + noise
def degrade_depth(lr):
    mask = np.random.rand(*lr.shape) > 0.6
    noisy_lr = lr * mask
    noisy_lr = noisy_lr + np.random.normal(0, 0.01, noisy_lr.shape)
    return noisy_lr

def main():
    files = sorted(glob.glob("samples/rgb_*.png"))
    print(f"Testing on {len(files)} samples with LiDAR Degradation...\n")

    results = {f"bicubic_x{s}": [] for s in SCALES}
    results.update({f"guided_x{s}": [] for s in SCALES})
    results.update({f"hybrid_x{s}": [] for s in SCALES})
    
    timings = {"bicubic": [], "guided": [], "hybrid": []}

    for f in files:
        idx = os.path.basename(f).split("_")[1].split(".")[0]
        rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        gt = np.load(f"samples/depth_{idx}.npy").astype(np.float32)
        H, W = gt.shape

        for s in SCALES:
            # 1. Get Low-Res Depth
            lr = downsample_depth(gt, s)
            
            # 2. Degrade it (Simulate sensor noise/holes)
            lr_degraded = degrade_depth(lr)

            # Bicubic on degraded data
            t = time.time()
            bic = bicubic_upsample(lr_degraded, (H, W))
            timings["bicubic"].append(time.time() - t)

            # Guided Filter on degraded data
            t = time.time()
            gf = guided_filter_upsample(rgb, lr_degraded, (H, W))
            timings["guided"].append(time.time() - t)

            # OUR HYBRID METHOD on degraded data
            t = time.time()
            hybrid, trust_map, _, _ = adaptive_fusion(rgb, lr_degraded, s)
            timings["hybrid"].append(time.time() - t)

            for name, pred in [(f"bicubic_x{s}", bic), (f"guided_x{s}", gf), (f"hybrid_x{s}", hybrid)]:
                m = compute_metrics(pred, gt)
                m["BoundaryF1"] = boundary_f1(pred, gt)
                m["EdgeRMSE"] = edge_rmse(pred, gt)
                results[name].append(m)

    print("===== RESULTS (Average over LiDAR-degraded samples) =====")
    for k, v in results.items():
        keys = v[0].keys()
        row = {key: np.nanmean([r[key] for r in v]) for key in keys}
        print(f"{k:12} | " + " | ".join([f"{key}: {row[key]:.4f}" for key in keys]))

    print("\nTime per image (ms):")
    print(f"Bicubic: {1000 * np.mean(timings['bicubic']):.2f} ms")
    print(f"Guided : {1000 * np.mean(timings['guided']):.2f} ms")
    print(f"Hybrid : {1000 * np.mean(timings['hybrid']):.2f} ms")

if __name__ == "__main__":
    main()
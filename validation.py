import matplotlib
matplotlib.use('Agg')  # <--- THIS FIXES THE ERROR (Runs headless)

import numpy as np
import cv2
import matplotlib.pyplot as plt
from hybrid import adaptive_fusion
import glob, os

# Load one sample
f = sorted(glob.glob("samples/rgb_*.png"))[0]
idx = os.path.basename(f).split("_")[1].split(".")[0]
rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
gt = np.load(f"samples/depth_{idx}.npy").astype(np.float32)
lr = cv2.resize(gt, (gt.shape[1]//4, gt.shape[0]//4), interpolation=cv2.INTER_AREA)

# Run Hybrid
fused, trust, _, _ = adaptive_fusion(rgb, lr, 4)

# Calculate Actual Error
error = np.abs(fused - gt)

# Flatten and bin by Trust
trust_flat = trust.flatten()
error_flat = error.flatten()

bins = np.linspace(0, 1, 11)
bin_centers = (bins[:-1] + bins[1:]) / 2
mean_errors = []

for i in range(len(bins)-1):
    mask = (trust_flat >= bins[i]) & (trust_flat < bins[i+1])
    if mask.sum() > 0:
        mean_errors.append(np.mean(error_flat[mask]))
    else:
        mean_errors.append(0)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(bin_centers, mean_errors, marker='o', linestyle='-', color='b')
plt.title("Trust Map Validation (Sparsification Curve)")
plt.xlabel("Trust Map Confidence Score")
plt.ylabel("Mean Absolute Error (MAE)")
plt.grid(True)
plt.savefig("trust_validation_plot.png")
print("Saved validation plot to trust_validation_plot.png")
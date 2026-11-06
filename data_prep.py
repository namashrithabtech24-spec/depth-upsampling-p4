import os
import numpy as np
from PIL import Image
from datasets import load_dataset

os.makedirs("samples", exist_ok=True)

print("Downloading NYU Depth V2 subset... (This may take 2 minutes)")
# We only load the validation set to keep it fast
ds = load_dataset("sayakpaul/nyu_depth_v2", split="validation")

print(f"Found {len(ds)} total images. Saving 30 for our test.")
for i in range(min(30, len(ds))):
    item = ds[i]
    rgb = item["image"].convert("RGB")
    depth = np.array(item["depth"]).astype(np.float32)

    if depth.max() > depth.min():
        depth = (depth - depth.min()) / (depth.max() - depth.min())

    rgb.save(f"samples/rgb_{i:03d}.png")
    np.save(f"samples/depth_{i:03d}.npy", depth)
print("Data prep complete! Check the 'samples' folder.")
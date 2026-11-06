import torch
import cv2
import numpy as np
import os

print("Loading MiDaS AI model (takes 10 seconds)...")
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model.eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

os.makedirs("samples", exist_ok=True)

for i in range(1, 4):
    filename = f"img{i}.jpg"
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found! Please download 3 images and rename them.")
        continue
        
    print(f"Processing {filename}...")
    img = cv2.imread(filename)
    if img is None:
        print(f"Error reading {filename}. Is it a valid image?")
        continue
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to standard size for speed
    img = cv2.resize(img, (640, 480))
    
    # Generate pseudo-GT depth
    inp = transform(img)
    with torch.no_grad():
        pred = model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=img.shape[:2],
            mode="bicubic", align_corners=False).squeeze().cpu().numpy()
            
    # Normalize to 0-1
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    
    # Save exactly how baseline.py expects them
    idx = f"{i-1:03d}"
    cv2.imwrite(f"samples/rgb_{idx}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    np.save(f"samples/depth_{idx}.npy", pred.astype(np.float32))

print("Data generation complete! Check the 'samples' folder.")
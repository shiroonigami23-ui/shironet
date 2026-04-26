import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path

root = Path(__file__).resolve().parents[1]
assets = root / "docs" / "assets"
assets.mkdir(parents=True, exist_ok=True)

# PNG chart
x = np.arange(1, 11)
train = np.exp(-x / 7) + 0.05 * np.sin(x)
adv = train + 0.07

plt.figure(figsize=(8, 4.5))
plt.plot(x, train, label="Baseline Loss", linewidth=2)
plt.plot(x, adv, label="Adversarial Loss", linewidth=2)
plt.title("ShiroNet Distillation Progress (Synthetic)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(assets / "training_curve.png", dpi=140)
plt.close()

# GIF animation
frames = []
W, H = 500, 220
for i in range(12):
    img = Image.new("RGB", (W, H), (20, 27, 38))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(30, 40), (470, 180)], outline=(120, 180, 255), width=2)
    progress = 30 + int((440 * (i + 1)) / 12)
    draw.rectangle([(30, 140), (progress, 180)], fill=(73, 176, 255))
    draw.text((35, 15), f"ShiroNet Model Evolution Step {i+1}/12", fill=(220, 230, 245))
    draw.text((35, 105), "Distillation + Adversarial Hardening", fill=(180, 210, 240))
    frames.append(img)

frames[0].save(
    assets / "model_evolution.gif",
    save_all=True,
    append_images=frames[1:],
    duration=160,
    loop=0,
)

print("Assets generated in docs/assets")

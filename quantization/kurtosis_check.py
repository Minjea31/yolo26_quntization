import os
import numpy as np
import torch
import cv2
from scipy.stats import kurtosis
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

CALIB_DIR = "../dataset/train/images"
N = 200  # number of calibration images

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = YOLO("../model/best.pt").model
model.to(device)
model.eval()

lb = LetterBox((640, 640), auto=False, stride=32)

def load_image(path):
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(path)
    im = im[:, :, ::-1]  # BGR -> RGB
    # LetterBox may return a dict with key 'image' or the image array directly.
    # Let failures propagate: the caller skips the image rather than feeding a
    # non-resized frame that would mix input sizes and pollute the stats.
    res = lb(image=np.ascontiguousarray(im))
    img = res["image"] if isinstance(res, dict) else res
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.0)
    return t.unsqueeze(0).to(device)


stats = {}  # name -> list of dict per image


def mk_hook(name):
    def hook(mod, inp, out):
        try:
            x = inp[0].detach().float().cpu().numpy().ravel()
        except Exception:
            return
        s = stats.setdefault(name, [])
        p_lo, p_hi = np.percentile(x, [0.1, 99.9])
        kt = kurtosis(x)
        if not np.isfinite(kt):
            # dead channel after pruning -> std ~ 0 -> kurtosis is nan; ignore it
            kt = 0.0
        s.append({
            "kurt": float(kt),
            "prange": float(p_hi - p_lo),
            "absmax": float(np.abs(x).max()),
            "std": float(x.std()),
        })

    return hook


handles = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        handles.append(module.register_forward_hook(mk_hook(name)))

imgs = [f for f in sorted(os.listdir(CALIB_DIR)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
imgs = imgs[:N]

with torch.no_grad():
    for k, f in enumerate(imgs):
        path = os.path.join(CALIB_DIR, f)
        try:
            model(load_image(path))
        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue
        if (k + 1) % 50 == 0:
            print(f"Processed {k+1}/{len(imgs)} images")

for h in handles:
    h.remove()

rows = []
for name, lst in stats.items():
    if not lst:
        continue
    mean_stats = {k: np.mean([d[k] for d in lst]) for k in lst[0]}
    absmax_global = max(d["absmax"] for d in lst)  # true INT8 clipping concern
    rows.append((name, mean_stats["kurt"], mean_stats["prange"], mean_stats["absmax"], absmax_global, mean_stats["std"]))

print(f"\n{'layer':45s} {'kurt':>8s} {'prange':>8s} {'absmax':>8s} {'amax_g':>8s} {'std':>7s}")
for name, kt, pr, am, amg, sd in sorted(rows, key=lambda x: x[0]):
    print(f"{name:45s} {kt:8.1f} {pr:8.2f} {am:8.2f} {amg:8.2f} {sd:7.3f}")

mean_stats_dict = {name: {"kurt": float(np.mean([d['kurt'] for d in lst])),
                         "prange": float(np.mean([d['prange'] for d in lst])),
                         "absmax": float(np.mean([d['absmax'] for d in lst])),
                         "absmax_global": float(max(d['absmax'] for d in lst)),
                         "std": float(np.mean([d['std'] for d in lst]))}
                   for name, lst in stats.items() if lst}

np.save("calib_actstats.npy", mean_stats_dict, allow_pickle=True)
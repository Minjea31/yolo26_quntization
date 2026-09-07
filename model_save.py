from pathlib import Path
from ultralytics import YOLO
import torch.nn as nn

# 저장 폴더 생성
save_dir = Path("./prune_info")
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "model_structure.txt"

# 모델 로드
model = YOLO("yolo26n.pt").model

with open(save_path, "w", encoding="utf-8") as f:
    f.write("=" * 120 + "\n")
    f.write("YOLO26n Model Structure\n")
    f.write("=" * 120 + "\n\n")

    for name, module in model.named_modules():
        num_params = sum(p.numel() for p in module.parameters(recurse=False))

        f.write(f"[{name if name else 'root'}]\n")
        f.write(f"Type       : {type(module).__name__}\n")
        f.write(f"Parameters : {num_params:,}\n")

        # Conv 계열은 weight shape도 저장
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            for pname, p in module.named_parameters(recurse=False):
                f.write(f"  {pname:<10}: {tuple(p.shape)}\n")

        f.write("-" * 120 + "\n")

print(f"Saved to: {save_path}")

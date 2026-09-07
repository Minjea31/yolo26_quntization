from ultralytics import YOLO
from compress.Compress import PruneHandler as ph
import argparse
from pathlib import Path
import math

parser = argparse.ArgumentParser()
parser.add_argument('--bmodel', type=str, default='/home/a/yolo26_pruning/best.pt')
parser.add_argument('--pruning_ratio', type=float, default='0.5') # 몇 %를 자를건지
parser.add_argument('--prune_type', type=str, default='ALL', help="H or B or ALL")
parser.add_argument('--method', type=str, default='GM', help="L1 or L2 or GM")
parser.add_argument('--cfg_output_path', type=str, default='prune')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--name', type=str, default='prune_model')
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--resume_path', type=str)
parser.add_argument('--align', type=int, default=8,
                    help="남길 채널 수를 이 값의 배수로 맞춤(양자화/NPU 정렬용). 1이면 정렬 없음.")
parser.add_argument("--device", type=str, default='0')
parser.add_argument('--data', type=str, default=str(Path(__file__).resolve().parents[1] / 'dataset.yaml'))

args = parser.parse_args()
project_dir = Path(__file__).resolve().parent / "checkpoints"
pruning_ratio = 1 - math.sqrt(1 - args.pruning_ratio) # in과 out을 다 자르기 때문에 a^2 형태로 비율이 들어가게 됨.

if not args.resume_path:
    bigmodel = YOLO(args.bmodel)
    ph = ph(bigmodel, pruning_ratio, args.method, args.cfg_output_path, args.prune_type, align=args.align)
    cmodel = ph.compress_yolo26()

    cmodel.train(data=args.data, epochs=args.epoch, imgsz=640, resume=False,
                 device=args.device, name=args.name, batch=args.bs, workers=4,
                 project=str(project_dir))
else:
    cmodel = YOLO(args.resume_path)
    cmodel.train(data=args.data, epochs=50, imgsz=640, device=args.device, name=args.name, batch=args.bs, workers=4,
                 resume=True, project=str(project_dir))

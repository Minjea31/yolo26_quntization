"""Test whether QuantFormat.QOperator gets more Conv layers onto real INT8 kernels
than the default QDQ format, since QDQ fusion is blocked by ORT's internal
'single consumer' requirement (see README: INT8 커널 선택 분석) regardless of
channel count / pruning ratio.

Usage:
    cd yolo26
    python3 ../quantization/qoperator_test.py --model ../model/baseline.pt --data ../dataset.yaml
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import onnx

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="../model/baseline.pt")
parser.add_argument("--data", type=str, default="../dataset.yaml")
parser.add_argument("--imgsz", type=int, default=640)
args = parser.parse_args()

from ultralytics import YOLO

m = YOLO(args.model)
fp32_onnx = m.export(format="onnx", imgsz=args.imgsz, opset=17, simplify=False)
fp32_onnx = Path(fp32_onnx)
print(f"FP32 ONNX: {fp32_onnx}")

raw = onnx.load(str(fp32_onnx))
n_conv = sum(1 for n in raw.graph.node if n.op_type == "Conv")
print(f"raw FP32 graph Conv count: {n_conv}")

from ultralytics.utils.export.onnx import onnx_calibration_reader
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

import numpy as np
import torch
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset


def build_calib_loader():
    data = check_det_dataset(args.data)
    cfg = get_cfg()
    cfg.imgsz = args.imgsz
    ds = build_yolo_dataset(cfg, data.get("val", data.get("train")), 1, data, mode="val", rect=False)
    return build_dataloader(ds, batch=1, workers=0, shuffle=False)


def transform_fn(data_item):
    x = data_item["img"] if isinstance(data_item, dict) else data_item
    assert x.dtype == torch.uint8
    im = x.numpy().astype(np.float32) / 255.0
    return im[None] if im.ndim == 3 else im


loader = build_calib_loader()

# 속도 우선 전략: model.23(Detect head의 decode/후처리 블록 전체 - Conv +
# Sigmoid/TopK/GatherElements/Concat 등 146개 노드)만 통째로 FP32로 남기고,
# backbone/neck(model.0~22)은 Conv/Sigmoid/Mul/Concat 가리지 않고 전부 양자화한다.
# (Conv만 남기고 나머지 op은 다 빼는 방식은 backbone까지 파편화시켜 오히려 느려짐 - 20절 참고)
raw_graph_nodes = onnx.load(str(fp32_onnx)).graph.node
nodes_to_exclude = sorted(n.name for n in raw_graph_nodes if n.name.startswith("/model.23"))
print(f"quantization에서 제외할 model.23(Detect head 전체) 노드: {len(nodes_to_exclude)}개")

cases = [
    ("QDQ (baseline, 기존 방식, S8/S8)", QuantFormat.QDQ, QuantType.QInt8),
    ("QOperator (S8 activation, x64 비권장)", QuantFormat.QOperator, QuantType.QInt8),
    ("QOperator + U8 activation (권장 조합)", QuantFormat.QOperator, QuantType.QUInt8),
]

for fmt_name, fmt, act_type in cases:
    out_path = fp32_onnx.with_name(f"{fp32_onnx.stem}_{fmt.name.lower()}_{act_type.name.lower()}.onnx")
    quantize_static(
        str(fp32_onnx),
        str(out_path),
        onnx_calibration_reader(loader, transform_fn, batch=0),
        quant_format=fmt,
        activation_type=act_type,
        nodes_to_exclude=nodes_to_exclude,
    )
    q = onnx.load(str(out_path))
    c = Counter(n.op_type for n in q.graph.node)
    print(f"\n=== {fmt_name} -> {out_path.name} ===")
    print(f"QLinearConv: {c.get('QLinearConv', 0)}  (진짜 INT8 커널, quantize_static 시점에 바로 결정됨)")
    print(f"Conv(FP32 폴백): {c.get('Conv', 0)}")
    print(f"DequantizeLinear: {c.get('DequantizeLinear', 0)}  QuantizeLinear: {c.get('QuantizeLinear', 0)}")

"""Build a TensorRT-compatible QDQ INT8 ONNX (symmetric quantization, INT32 bias
not wrapped in Q/DQ) with model.23 (Detect head decode block) excluded entirely.

TensorRT's ONNX parser rejects the CPU-optimized QDQ file from qoperator_test.py
because it uses asymmetric activation quantization (nonzero zero_point) - fine for
ORT's CPU VNNI kernels, but TensorRT only accepts symmetric INT8 (zero_point=0).
See "Quantization 개념 정리" Notion page §14.2 / §17.4 for why.

Usage:
    cd yolo26
    python3 ../quantization/export_trt_qdq.py --model ../model/baseline.pt --data ../dataset.yaml
"""

import argparse
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

raw_graph_nodes = onnx.load(str(fp32_onnx)).graph.node
# model.23(Detect head 전체) + C2PSA attention(model.10/model.22의 /attn/ 아래 전부) 제외.
# attention 내부의 0-d 스칼라 상수(예: 1/sqrt(d) 같은 고정 계수)를 quantize_static이 양자화하려다
# TensorRT 파서에서 axis 에러가 남 (기존 yolo26_TensorRT 프로젝트의 Tier 2 exclude 리스트와 동일한 이유).
nodes_to_exclude = sorted(
    n.name for n in raw_graph_nodes if n.name.startswith("/model.23") or "/attn/" in n.name
)
print(f"quantization에서 제외할 노드 (model.23 + attention): {len(nodes_to_exclude)}개")

out_path = fp32_onnx.with_name(f"{fp32_onnx.stem}_qdq_qint8_trt.onnx")
quantize_static(
    str(fp32_onnx),
    str(out_path),
    onnx_calibration_reader(loader, transform_fn, batch=0),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,  # attention을 nodes_to_exclude로 뺐으니 나머지 Conv는 per-channel로 정확도 챙김
    nodes_to_exclude=nodes_to_exclude,
    extra_options={
        "ActivationSymmetric": True,  # TensorRT: zero_point must be 0
        "WeightSymmetric": True,
        "QuantizeBias": False,  # TRT 10 rejects DequantizeLinear on INT32 bias
    },
)

q = onnx.load(str(out_path))
c = Counter(n.op_type for n in q.graph.node)
print(f"\n=== TRT-compatible QDQ -> {out_path.name} ===")
print(f"QuantizeLinear: {c.get('QuantizeLinear', 0)}  DequantizeLinear: {c.get('DequantizeLinear', 0)}")
print(f"Conv (양자화 대상 + model.23 FP32 둘 다 포함, raw graph 그대로): {c.get('Conv', 0)}")

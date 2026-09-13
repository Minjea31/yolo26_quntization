"""Compare INT8 ONNX variants (QDQ vs QOperator, S8 vs U8 activation, FP32 baseline)
side by side: real-kernel fusion rate, end-to-end latency, and (optionally) mAP.

kernel_check.py / qoperator_test.py answer these one model / one aspect at a time;
this script puts them in one table so the QDQ vs QOperator tradeoff is visible at a glance.

Usage:
    cd quantization
    python3 compare_formats.py
    python3 compare_formats.py --map --data ../dataset.yaml   # mAP까지 같이 측정 (느림)
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument(
    "--models",
    type=str,
    nargs="+",
    default=[
        "../model/baseline.onnx",
        "../model/baseline_qdq_qint8.onnx",
        "../model/baseline_qoperator_qint8.onnx",
        "../model/baseline_qoperator_quint8.onnx",
    ],
)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--provider", type=str, default="CPUExecutionProvider")
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--iters", type=int, default=100)
parser.add_argument("--map", action="store_true", help="ultralytics val()로 mAP도 같이 측정 (모델당 val set 전체 순회, 느림)")
parser.add_argument("--data", type=str, default="../dataset.yaml")
parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
parser.add_argument("--batch", type=int, default=2)
parser.add_argument(
    "--trt-cache-dir",
    type=str,
    default="../model/trt_cache",
    help="TensorrtExecutionProvider 엔진 캐시 경로 (재실행 시 재빌드 방지, 첫 빌드는 모델당 수십 초~수 분 걸릴 수 있음)",
)
args = parser.parse_args()

# QOperator(QLinearConv 등)는 onnxruntime 전용 커스텀 op라 TensorRT 파서가 못 읽는다.
# GPU(Tensorrt/CUDA) provider일 때 기본 모델 목록을 TRT가 실제로 이해하는 QDQ 포맷으로 좁힌다.
if args.provider in ("TensorrtExecutionProvider", "CUDAExecutionProvider") and args.models == parser.get_default("models"):
    # baseline_qdq_qint8.onnx는 CPU VNNI용 비대칭 양자화라 TRT가 거부함 (symmetric만 허용).
    # export_trt_qdq.py로 만든 대칭 양자화 버전을 대신 사용.
    args.models = ["../model/baseline.onnx", "../model/baseline_qdq_qint8_trt.onnx"]
    print(f"[info] provider={args.provider} -> QOperator/비대칭 QDQ는 TRT가 못 읽어서 기본 모델 목록을 대칭 QDQ로 좁힘: {args.models}")


def real_int8_kernel_count(graph):
    return sum(1 for n in graph.node if n.op_type.startswith("QLinear") or n.op_type == "QGemm")


def build_providers(path: Path):
    if args.provider == "TensorrtExecutionProvider":
        is_quantized = "int8" in path.stem.lower() or "qdq" in path.stem.lower()
        cache_dir = Path(args.trt_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        trt_opts = {
            # baseline(FP32)은 순수 FP16 엔진(= yolo26_TensorRT의 best_fp16.engine과 동급) 재현용.
            # QDQ 파일은 그래프의 Q/DQ가 가리키는 부분만 INT8, 나머지(model.23 등)는 fp16_enable 덕에 FP16으로.
            "trt_fp16_enable": True,
            "trt_int8_enable": is_quantized,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache_dir),
        }
        return [("TensorrtExecutionProvider", trt_opts), "CPUExecutionProvider"]
    return [args.provider]


def profile_model(path: Path):
    raw = onnx.load(str(path))
    providers = build_providers(path)
    is_cpu_ep = args.provider == "CPUExecutionProvider"

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt = None
    if is_cpu_ep:
        # CPU EP의 QDQ->QLinearConv fusion 통계용. TRT/CUDA는 그래프를 통째로 EP가 삼켜서
        # ONNX 레벨 fusion 카운트가 의미 없음 (TRT 내부 엔진은 ONNX 노드로 안 드러남).
        optimized_path = str(path.with_name(path.stem + "_cmp_optimized.onnx"))
        so.optimized_model_filepath = optimized_path
    sess = ort.InferenceSession(str(path), sess_options=so, providers=providers)
    if is_cpu_ep:
        opt = onnx.load(optimized_path)

    print(f"    실제 활성 provider: {sess.get_providers()}")
    input_name = sess.get_inputs()[0].name
    input_dtype = np.uint8 if "uint8" in str(sess.get_inputs()[0].type) else np.float32
    if input_dtype == np.uint8:
        dummy = np.random.randint(0, 256, (1, 3, args.imgsz, args.imgsz), dtype=np.uint8)
    else:
        dummy = np.random.rand(1, 3, args.imgsz, args.imgsz).astype(np.float32)

    for _ in range(args.warmup):
        sess.run(None, {input_name: dummy})

    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)

    return {
        "raw_ops": Counter(n.op_type for n in raw.graph.node),
        "opt_ops": Counter(n.op_type for n in opt.graph.node) if opt is not None else None,
        "n_conv_raw": sum(1 for n in raw.graph.node if n.op_type == "Conv"),
        "n_int8_kernel": real_int8_kernel_count(opt.graph) if opt is not None else None,
        "n_fp32_fallback_conv": sum(1 for n in opt.graph.node if n.op_type == "Conv") if opt is not None else None,
        "latency_mean_ms": times.mean(),
        "latency_p50_ms": np.percentile(times, 50),
        "latency_p99_ms": np.percentile(times, 99),
    }


def eval_map(path: Path):
    from ultralytics import YOLO

    model = YOLO(str(path))
    metrics = model.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch)
    return metrics.box.map, metrics.box.map50


rows = []
for model_str in args.models:
    path = Path(model_str)
    if not path.exists():
        print(f"[skip] {path} 없음")
        continue
    print(f"\n=== profiling {path.name} ===")
    stats = profile_model(path)
    row = {"model": path.name, **stats}
    if args.map:
        print(f"    mAP 측정 중 (split={args.split}) ...")
        row["map50-95"], row["map50"] = eval_map(path)
    rows.append(row)

header = ["model", "raw Conv", "INT8 kernel(QLinear*)", "FP32 fallback Conv", "latency mean(ms)", "p50(ms)", "p99(ms)"]
if args.map:
    header += ["mAP50-95", "mAP50"]

print("\n" + " | ".join(header))
print("|".join(["---"] * len(header)))
for r in rows:
    line = [
        r["model"],
        str(r["n_conv_raw"]),
        str(r["n_int8_kernel"]) if r["n_int8_kernel"] is not None else "n/a (TRT 내부)",
        str(r["n_fp32_fallback_conv"]) if r["n_fp32_fallback_conv"] is not None else "n/a (TRT 내부)",
        f"{r['latency_mean_ms']:.3f}",
        f"{r['latency_p50_ms']:.3f}",
        f"{r['latency_p99_ms']:.3f}",
    ]
    if args.map:
        line += [f"{r['map50-95']:.4f}", f"{r['map50']:.4f}"]
    print(" | ".join(line))

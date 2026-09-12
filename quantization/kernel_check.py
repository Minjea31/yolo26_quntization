import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="../model/baseline_int8.onnx")
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--provider", type=str, default="CPUExecutionProvider")
args = parser.parse_args()

raw = onnx.load(args.model)
print("=== raw QDQ graph op_type counts (quantize_static 직후) ===")
print(Counter(n.op_type for n in raw.graph.node))

optimized_path = str(Path(args.model).with_name(Path(args.model).stem + "_optimized.onnx"))

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.optimized_model_filepath = optimized_path
so.log_severity_level = 0  # VERBOSE: 커널/provider 배정 로그가 stderr로 출력됨
so.enable_profiling = True

sess = ort.InferenceSession(args.model, sess_options=so, providers=[args.provider])

opt = onnx.load(optimized_path)
print(f"\n=== optimized graph op_type counts (provider={args.provider}, ORT가 실제로 fusion한 결과) ===")
print(Counter(n.op_type for n in opt.graph.node))

input_name = sess.get_inputs()[0].name
dummy = np.random.rand(1, 3, args.imgsz, args.imgsz).astype(np.float32)
sess.run(None, {input_name: dummy})
trace_path = sess.end_profiling()
print(f"\nprofiling trace saved to {trace_path}")

with open(trace_path) as f:
    events = json.load(f)
node_events = [e for e in events if e.get("cat") == "Node" and "op_name" in e.get("args", {})]

print("\n=== 실제 실행된 노드의 op_type 분포 (profiling trace 기준) ===")
print(Counter(e["args"]["op_name"] for e in node_events))

print("\n=== 실제 실행된 노드의 provider 분포 ===")
print(Counter(e["args"].get("provider", "?") for e in node_events))

print("\n=== op_type별 누적 실행 시간(us) 상위 15개 ===")
dur_by_op = Counter()
for e in node_events:
    dur_by_op[e["args"]["op_name"]] += e.get("dur", 0)
for op, dur in dur_by_op.most_common(15):
    print(f"{op:30s} {dur:>10d} us")

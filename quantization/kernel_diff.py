"""Compare a QDQ INT8 ONNX graph against its ORT-optimized graph to see which Conv
layers actually got fused into a native INT8 kernel (QLinearConv) and which stayed
as plain FP32 Conv wrapped by DequantizeLinear/QuantizeLinear.

Usage (run after kernel_check.py has produced the "_optimized.onnx" file):
    python3 kernel_diff.py --raw ../model/baseline_int8.onnx \
                            --optimized ../model/baseline_int8_optimized.onnx
"""

import argparse
from collections import Counter, defaultdict

import onnx
from onnx import numpy_helper

parser = argparse.ArgumentParser()
parser.add_argument("--raw", type=str, default="../model/baseline_int8.onnx")
parser.add_argument("--optimized", type=str, default="../model/baseline_int8_optimized.onnx")
args = parser.parse_args()

raw = onnx.load(args.raw)
opt = onnx.load(args.optimized)

# Nodes that survive as plain "Conv" in the optimized graph keep their original name;
# nodes fused into QLinearConv get a brand-new synthetic name, so "name disappeared
# from the optimized Conv set" is what identifies a fused layer.
opt_conv_names = {n.name for n in opt.graph.node if n.op_type == "Conv"}
raw_convs = [n for n in raw.graph.node if n.op_type == "Conv"]

producer = {}
for n in raw.graph.node:
    for o in n.output:
        producer[o] = n

init = {i.name: numpy_helper.to_array(i) for i in raw.graph.initializer}


def get_attr(node, key, default=None):
    for a in node.attribute:
        if a.name == key:
            if a.type == onnx.AttributeProto.INTS:
                return list(a.ints)
            if a.type == onnx.AttributeProto.INT:
                return a.i
    return default


def zp_scale(dq_node):
    if dq_node is None or dq_node.op_type != "DequantizeLinear":
        return None, None
    zp = init.get(dq_node.input[2]) if len(dq_node.input) > 2 else None
    scale = init.get(dq_node.input[1])
    return zp, scale


rows = []
for n in raw_convs:
    fused = n.name not in opt_conv_names
    block = n.name.split("/")[1] if n.name.startswith("/") else "?"  # e.g. "model.10"
    act_dq = producer.get(n.input[0])
    act_zp, act_scale = zp_scale(act_dq)
    rows.append(
        {
            "name": n.name,
            "fused": fused,
            "block": block,
            "group": get_attr(n, "group", 1),
            "kernel_shape": get_attr(n, "kernel_shape"),
            "act_zero_point": int(act_zp) if act_zp is not None and act_zp.size == 1 else act_zp,
            "act_scale": float(act_scale) if act_scale is not None and act_scale.size == 1 else act_scale,
        }
    )

fused_rows = [r for r in rows if r["fused"]]
unfused_rows = [r for r in rows if not r["fused"]]

print(f"=== 총 Conv {len(rows)}개 중 QLinearConv로 fusion된 것: {len(fused_rows)}개, 미fusion(FP32 폴백): {len(unfused_rows)}개 ===\n")

print("--- fusion된 레이어 (block별 카운트) ---")
print(Counter(r["block"] for r in fused_rows))

print("\n--- fusion된 레이어 상세 ---")
for r in fused_rows:
    print(f"  {r['name']:55s} group={r['group']} kernel={r['kernel_shape']} act_zp={r['act_zero_point']} act_scale={r['act_scale']}")

print("\n--- 미fusion 레이어가 속한 block 분포 (상위 10) ---")
print(Counter(r["block"] for r in unfused_rows).most_common(10))

print("\n--- fused vs unfused: activation zero_point 분포 비교 ---")
fused_zp = [r["act_zero_point"] for r in fused_rows if isinstance(r["act_zero_point"], int)]
unfused_zp = [r["act_zero_point"] for r in unfused_rows if isinstance(r["act_zero_point"], int)]
print("fused act_zp:  min={}, max={}, mean={:.1f}".format(min(fused_zp), max(fused_zp), sum(fused_zp) / len(fused_zp)))
print("unfused act_zp: min={}, max={}, mean={:.1f}".format(min(unfused_zp), max(unfused_zp), sum(unfused_zp) / len(unfused_zp)))

print(
    "\n=== 근본 원인 (gdb 라이브 디버깅으로 확정, README 참고) ===\n"
    "ConvNodeGroupSelector::Check() 자체의 조건은 fused/unfused 전부 동일하게 통과합니다.\n"
    "진짜 원인은 그 앞 단계인 QDQS8ToU8Transformer(qdq_s8_to_u8.cc)의 CheckOutputEdges 체크입니다:\n"
    "CPU 커널은 UINT8(U8) 기준인데 quantize_static 기본값은 부호있는 INT8(S8)라, fusion 전에\n"
    "S8->U8 승격을 시도하는데, 이 승격은 'Q 노드의 소비자가 정확히 1개일 때만' 적용됩니다.\n"
    "\n"
    "ONNX Runtime을 --config Debug로 직접 빌드해 qdq_s8_to_u8.cc:95에 브레이크포인트를 걸고\n"
    "실제 /model.0/conv/Conv의 출력 Q 노드에서 멈춰 확인한 결과:\n"
    "  CheckOutputEdges(graph, node, 1) == false\n"
    "  node.GetOutputEdgesCount() == 2   (raw quantize_static 출력 파일에서는 1이었음!)\n"
    "\n"
    "즉 ORT가 세션 로드 중 먼저 실행하는 Level 1 최적화 패스 중 하나가, quantize_static이 만든\n"
    "원본 그래프에는 없던 두 번째 소비자 엣지를 이 텐서에 추가합니다. raw ONNX 파일을 파이썬으로\n"
    "정적으로 아무리 뜯어봐도 이건 안 보이고, gdb로 실행 중인 프로세스를 직접 들여다봐야만\n"
    "확인됩니다. fusion된 12개(C2PSA attn 블록, Detect head one2one 최종 1x1 conv)는 이런 추가\n"
    "엣지가 안 생기는 분기 없는 직선 체인이라 fusion에 성공한 것으로 보입니다."
)

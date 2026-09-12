# QDQ 포맷과 INT8 fusion — 정리

> YOLO26n baseline 모델을 INT8로 양자화했는데, Conv 102개 중 12개만 진짜 INT8 커널(`QLinearConv`)로 바뀌고 나머지 90개는 그대로 FP32로 돈 이유를 파헤친 정리.

---

## 1. QDQ 포맷이 하는 일

`quantize_static(..., quant_format=QuantFormat.QDQ)` (ONNX Runtime 기본값)는 그래프에 **양자화 파라미터(scale/zero_point)만 삽입**한다. Conv 자체는 안 바뀐다.

```
DQ (int8 → fp32 복원)  →  Conv (FP32 연산)  →  Q (fp32 → int8 재양자화)
```

**FP16이 아니라 FP32이고, 순서는 DQ가 먼저다** (Q가 먼저가 아니다).

raw `quantize_static` 출력 파일을 직접 열어보면 `QLinearConv`(진짜 INT8 커널)가 **0개**다 — 이 포맷은 애초에 양자화 시점에 커널을 결정할 생각이 없다.

## 2. 왜 이렇게 설계했나

`DQ → Conv(FP32) → Q`는 **INT8 커널이 하나도 없는 엔진에서도 항상 정확하게 실행되는 "안전한 기본 경로"**다. 그냥 FP32 수학이라 어디서 돌려도 답이 맞는다.

반면 `QLinearConv`는 그 dtype 조합의 INT8 커널이 실제로 구현돼 있어야만 동작한다 — 없으면 에러.

→ QDQ 전략 = **"일단 항상 돌아가게 내보내고, 실행 엔진(ONNX Runtime)이 알아서 눈치껏 빠른 INT8 커널로 바꿔치기(fusion)할 기회를 준다."**
fusion은 **필수가 아니라 선택적 최적화**다.

## 3. 진짜 커널 결정은 ORT가 세션을 "로드"할 때 일어남

`InferenceSession(...)`으로 QDQ 파일을 열면, ORT가 그래프 최적화 패스를 돌리며 `DQ→Conv→Q` 패턴을 `QLinearConv`로 바꿀지 노드마다 판단한다. 조건은 2단계.

### 3-1) `QDQS8ToU8Transformer` (먼저 실행)

x86 CPU의 빠른 INT8 커널은 **UINT8 activation** 기준인데, `quantize_static` 기본값은 **부호 있는 INT8(S8)**. 그래서 fusion 전에 S8→U8 승격을 먼저 시도한다.

> **승격 조건: 그 텐서를 만드는 Q 노드의 소비자(consumer)가 정확히 1개일 때만** (`CheckOutputEdges(graph, node, 1)`)

### 3-2) `ConvNodeGroupSelector::Check` (그다음)

> Conv에 **들어가는** activation dtype과 **나가는** dtype이 같아야 함 (`dt_input != dt_output` → fusion 거부)

## 4. 우리 모델에서 왜 끊겼나

YOLO26n의 backbone/neck은 **C3k2 split-concat, SPPF, skip connection**이 많아서, 한 Conv의 출력이 다음 레이어뿐 아니라 다른 브랜치에서도 동시에 읽힌다 — 소비자가 2개 이상인 텐서가 많다.

gdb로 실행 중인 프로세스에서 직접 확인:

```
(gdb) print optimizer_utils::CheckOutputEdges(graph, node, 1)
$3 = false
(gdb) print node.GetOutputEdgesCount()
$5 = 2      # raw 파일에서는 1이었는데, ORT 세션 로드 중 2로 늘어나 있었음
```

ORT가 로드 과정에서 먼저 실행하는 다른 최적화 패스가, quantize_static이 만든 원본 그래프엔 없던 두 번째 소비자 엣지를 추가한 것. **정적 파일 분석으로는 안 보이고, 실행 중 상태를 gdb로 봐야만 확인되는 부분이었다.**

결과 사슬:

```
1. CheckOutputEdges → false           (소비자 2개라서)
2. S8→U8 승격 실패                     → 이 Conv 출력 쪽은 INT8(S8)로 남음
3. 근데 입력 쪽은 (소비자 1개라) 성공적으로 UINT8 변환됨
4. 입력 U8 / 출력 S8 → dtype 불일치
5. ConvNodeGroupSelector::Check 실패   → fusion 최종 거부
6. → DequantizeLinear → Conv(FP32) → QuantizeLinear 로 확정
```

## 5. 왜 12개만 살아남았나

**C2PSA attention 블록**(`attn.pe→attn.proj→ffn.1`)과 **Detect head 최종 1x1 conv**(`one2one_cv2/cv3`)는 분기 없는 순수 직선 체인이라 다중 소비자 상황이 안 생긴다. 그래서 이 12곳만 입출력 양쪽 다 U8로 깨끗이 변환되고 fusion 성공.

**이건 채널 수(pruning ratio)와 무관하고, 순전히 그래프 위상(topology) 문제다.** → 채널을 줄이거나 늘려도 "누가 몇 곳에서 쓰이는가"라는 연결 구조 자체는 안 바뀐다.

## 6. "QDQ가 느리다"는 조건부로만 맞다

**fusion 성공한 12개**: 최적화되면서 `DQ`/`Q` 노드가 그래프에서 **아예 삭제되고** `QLinearConv` 하나로 교체된다. `QOperator`로 처음부터 만든 것과 최종 그래프가 완전히 동일 — 왕복 비용 없음.

**fusion 실패한 90개만** 실제로 `DQ→Conv(FP32)→Q`가 런타임에 그대로 실행된다. 실측:

| op | 누적 시간 |
|---|---|
| DequantizeLinear | 20,459us |
| Conv (FP32) | 14,797us |
| QuantizeLinear | 6,967us |
| **합계** | **42,223us** ← 이 90개가 부담 |
| QLinearConv (fusion된 12개) | 273us |

캐스팅 비용(27,426us)이 실제 Conv 연산(14,797us)보다 크다.

> **결론: QDQ 포맷 자체가 원래 느린 게 아니다.** QDQ와 QOperator는 fusion만 성공하면 완전히 동일한 최종 그래프로 수렴한다. 문제는 **이 모델의 branching 많은 구조 때문에 fusion이 90/102에서 실패**해서, 그 90개가 QDQ의 "안전하지만 느린 기본 경로"에 갇힌 것.

## 7. 용어: "그래프" = 구조

- **노드(node)** = 연산 하나 (Conv, Add, Concat, Split, QuantizeLinear ...)
- **엣지(edge)** = 노드 사이를 잇는 텐서 (데이터 흐름)
- **그래프 위상(topology)** = "어떤 레이어의 출력이 몇 개의 다음 레이어로 이어지는가" 같은 **연결 구조**

채널 개수·가중치 값은 그 구조 위에 얹힌 "크기/내용물"일 뿐, 구조 자체가 아니다. 그래서 채널을 조정해도(pruning) 이 fusion 문제는 해결되지 않는다.

## 8. 그래서 어떻게 하면 되나

`quant_format=QuantFormat.QOperator`를 쓰면 fusion 절차 자체가 필요 없어진다 — quantize_static 시점에 바로 `QLinearConv`로 박아 넣기 때문에 분기 구조와 무관하게 **100% 즉시 변환**된다 (실측: 102/102).

단, x64에서는 `QOperator` + signed INT8 activation 조합이 "커널은 진짜 INT8인데 최적화 안 된 느린 경로"일 수 있다는 ORT 자체 경고가 있음 →

```python
quantize_static(
    ...,
    quant_format=QuantFormat.QOperator,
    activation_type=QuantType.QUInt8,   # 권장: U8 activation
    weight_type=QuantType.QInt8,        # weight는 S8 유지
)
```

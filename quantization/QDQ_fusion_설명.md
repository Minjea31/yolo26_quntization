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

## 2. `QLinearConv`는 뭐하는 애인가

**INT8 값을 직접 받아서 INT8 채로 convolution을 계산하는 ONNX 연산자.** "진짜 INT8 커널"이라고 부른 게 바로 이 노드다.

일반 `Conv`는 입력이 2~3개(`x`, `w`, `b`, 전부 FP32)인데, `QLinearConv`는 **8~9개**다:

```
QLinearConv(
  x, x_scale, x_zero_point,      # 양자화된 입력 + 그 스케일/영점
  w, w_scale, w_zero_point,      # 양자화된 가중치 + 그 스케일/영점
  y_scale, y_zero_point,         # 출력을 다시 양자화할 스케일/영점
  bias                           # 있으면 INT32
)
→ y (INT8/UINT8)
```

**왜 하필 8~9개나 되나**: FP32 값은 값 자체가 이미 실수라서(`0.734`) 그대로 넘기면 끝이지만, INT8 값은 그냥 정수(`173`)일 뿐이라 `scale`/`zero_point` 없이는 이게 실제로 어떤 실수를 나타내는지 알 방법이 없다(`실수값 = scale × (173 − zero_point)`). 그래서 텐서 하나(`x`, `w`)마다 "정수값 + scale + zero_point" 3개씩을 세트로 넘겨야 하고, 출력을 다시 INT8로 양자화할 때 쓸 `y_scale`/`y_zero_point` 2개가 추가된다:

```
x(값) + x_scale + x_zero_point   = 3개  (입력)
w(값) + w_scale + w_zero_point   = 3개  (가중치)
y_scale + y_zero_point           = 2개  (출력을 다시 INT8로 만들 때 쓸 기준)
bias                             = 1개  (선택. 이미 INT32라 scale 별도로 안 필요)
─────────────────────────────────
합계 = 8개 (bias 없으면) ~ 9개 (있으면)
```

양자화 파라미터를 자기 안에 다 갖고 있어서, 입력부터 출력까지 **한 번에** 끝낸다.

내부 계산:

```
acc   = Σ (x_int - x_zero_point) × (w_int - w_zero_point)          # 정수 곱셈-누산, INT32로 쌓임
y_int = round(acc × (x_scale × w_scale) / y_scale) + y_zero_point   # 다시 INT8로 반올림
```

`x_int`, `w_int`가 이미 INT8이니 곱셈을 CPU/GPU의 **정수 전용 가속 명령어**(VNNI, DP4A, Tensor Core 등)로 바로 돌린다. 중간에 FP32로 안 풀린다.

### `DQ→Conv→Q` vs `QLinearConv` 비교

| | `DQ → Conv(FP32) → Q` | `QLinearConv` |
|---|---|---|
| 노드/커널 호출 | 3번 | **1번** |
| 중간 결과 | FP32 텐서를 메모리에 통째로 썼다가 다시 읽음 | 없음 (레지스터 안에서 끝) |
| 연산 종류 | float 곱셈 | int8 정수 곱셈 (하드웨어 가속) |

두 경로가 수학적으로는 거의 같은 결과를 내지만, `QLinearConv`는 세 단계를 한 커널 안에서 처리해서 중간 메모리 왕복이 통째로 사라진다.

## 3. 왜 QDQ는 이렇게(안전 경로 + 선택적 fusion) 설계했나

`DQ → Conv(FP32) → Q`는 **INT8 커널이 하나도 없는 엔진에서도 항상 정확하게 실행되는 "안전한 기본 경로"**다. 그냥 FP32 수학이라 어디서 돌려도 답이 맞는다.

반면 `QLinearConv`는 그 dtype 조합의 INT8 커널이 실제로 구현돼 있어야만 동작한다 — 없으면 에러.

→ QDQ 전략 = **"일단 항상 돌아가게 내보내고, 실행 엔진(ONNX Runtime)이 알아서 눈치껏 QLinearConv로 바꿔치기(fusion)할 기회를 준다."**
fusion은 **필수가 아니라 선택적 최적화**다.

## 4. 진짜 커널 결정은 ORT가 세션을 "로드"할 때 일어남

### 4.1 "로드"란 정확히 이 한 줄

```python
sess = ort.InferenceSession('model/baseline_int8.onnx', providers=['CPUExecutionProvider'])
```

이 줄이 실행되는 순간이 "세션 로드"다. 아직 `sess.run(...)`으로 추론을 돌리기 전이다.

### 4.2 이 한 줄 안에서 일어나는 일

1. **파일 읽기**: `.onnx`를 열어서 그래프 구조를 메모리에 올림 — 이 시점 그래프는 `quantize_static`이 만든 그대로 (`Conv`+`DQ`+`Q`뿐)
2. **그래프 최적화 패스를 순서대로 실행** — 로그에 찍히는 이런 줄들이 전부 이 안에서 일어남:
   ```
   GraphTransformer BiasGeluFusion modified: 0 with status: OK
   GraphTransformer NchwcTransformer modified: 1 with status: OK
   GraphTransformer NhwcTransformer modified: 1 with status: OK
   ```
3. 이 패스들 중 `QDQS8ToU8Transformer` → `ConvNodeGroupSelector`가 **"이 DQ→Conv→Q를 QLinearConv로 바꿀지"를 여기서 결정**하고, 결정되면 그래프를 메모리 상에서 실제로 고쳐 씀
4. `Node placements`, `Session successfully initialized`가 찍히면 — **이 시점에 그래프가 이미 최종 확정**됨

(gdb로 `QDQS8ToU8Transformer::ApplyImpl`에 브레이크포인트를 걸었을 때도, `sess.run()`을 부르기 전인 정확히 이 `InferenceSession(...)` 줄을 실행하는 동안 멈췄다.)

### 4.3 왜 이 타이밍이 중요한가

- **`quantize_static`**(파일을 만드는 도구)과 **`InferenceSession`**(그 파일을 로드해서 돌리는 엔진)은 **완전히 다른 두 단계**다.
- `.onnx` 파일 자체에는 "이 노드가 fusion됐다/안됐다"는 정보가 **없다**. 그 결정은 **`InferenceSession(...)`을 새로 호출할 때마다 그 자리에서 다시** 내려진다.
- 한 번 로드가 끝나면, `sess.run(...)`을 몇 번을 불러도 이미 확정된 그래프를 그대로 반복 실행할 뿐, 더 이상 "fusion할까?" 판단은 안 한다.

> 비유: `quantize_static`은 "설계도"만 그려주고, `InferenceSession(...)`이 그 설계도를 보고 "이 부분은 조립식 부품(QLinearConv)으로 갈아끼우자, 이건 원래대로 두자"를 **공장 가동 직전에 한 번** 정한다. 그 이후 실제 생산(`sess.run`)은 정해진 라인 그대로 계속 돈다.

### 4.4 조건 두 단계 (정리)

**4.4.1) `QDQS8ToU8Transformer` (먼저 실행)**

x86 CPU의 빠른 INT8 커널은 **UINT8 activation** 기준인데, `quantize_static` 기본값은 **부호 있는 INT8(S8)**. 그래서 fusion 전에 S8→U8 승격을 먼저 시도한다.

> 승격 조건: 그 텐서를 만드는 Q 노드의 소비자(consumer)가 **정확히 1개**일 때만 (`CheckOutputEdges(graph, node, 1)`)

**4.4.2) `ConvNodeGroupSelector::Check` (그다음)**

> Conv에 **들어가는** activation dtype과 **나가는** dtype이 같아야 함 (`dt_input != dt_output` → fusion 거부)

## 5. 우리 모델에서 왜 끊겼나

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

## 6. 왜 12개만 살아남았나

**C2PSA attention 블록**(`attn.pe→attn.proj→ffn.1`)과 **Detect head 최종 1x1 conv**(`one2one_cv2/cv3`)는 분기 없는 순수 직선 체인이라 다중 소비자 상황이 안 생긴다. 그래서 이 12곳만 입출력 양쪽 다 U8로 깨끗이 변환되고 fusion 성공.

**이건 채널 수(pruning ratio)와 무관하고, 순전히 그래프 위상(topology) 문제다.** → 채널을 줄이거나 늘려도 "누가 몇 곳에서 쓰이는가"라는 연결 구조 자체는 안 바뀐다.

## 7. "QDQ가 느리다"는 조건부로만 맞다

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

## 8. 용어: "그래프" = 구조

- **노드(node)** = 연산 하나 (Conv, Add, Concat, Split, QuantizeLinear ...)
- **엣지(edge)** = 노드 사이를 잇는 텐서 (데이터 흐름)
- **그래프 위상(topology)** = "어떤 레이어의 출력이 몇 개의 다음 레이어로 이어지는가" 같은 **연결 구조**

채널 개수·가중치 값은 그 구조 위에 얹힌 "크기/내용물"일 뿐, 구조 자체가 아니다. 그래서 채널을 조정해도(pruning) 이 fusion 문제는 해결되지 않는다.

## 9. 그래서 어떻게 하면 되나

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

## 10. GPU 관점 — CPU 얘기가 왜 계속 나왔나

이 문서에 나온 `MLAS`, `x86 VNNI`, `QDQS8ToU8Transformer` 등은 전부 **CPU Execution Provider 내부** 동작이다. GPU로 돌리고 싶어도 실제로 확인해보면:

- `providers=['CUDAExecutionProvider']`로 요청해도, 이 QDQ 양자화 그래프의 연산자 조합에 대해 **CUDA EP가 커널을 갖고 있지 않아** 전체 그래프가 조용히 CPU로 폴백됨 (`All nodes placed on [CPUExecutionProvider]`)
- `providers=['TensorrtExecutionProvider']`도 시스템에 TensorRT 라이브러리(`libnvinfer.so`)가 실제로 설치돼 있지 않으면 동일하게 CPU로 폴백됨

**즉 순수 `onnxruntime` 패키지로는 이 INT8 모델이 GPU에서 안 돈다.** 진짜 GPU INT8 실행은 TensorRT SDK를 따로 설치해서 TensorRT EP(또는 TensorRT 엔진 직접 빌드)를 써야 하며, 이게 이 프로젝트가 처음부터 TensorRT를 배포 경로로 택한 이유다.

## 11. 왜 90개나 됐나 — YOLO 백본 자체가 분기투성이 구조

"소비자 1개"라는 조건이 이렇게까지 많이(90/102, 약 88%) 걸린 게 이상해 보일 수 있지만, YOLO26n 백본 설계 자체가 원인이다.

`model.2`, `model.4`, `model.6`, `model.8` 같은 백본 블록은 전부 **C3k2**(CSP 계열) 블록인데:

```
입력 → cv1 → [반으로 split] → 한쪽은 그대로, 한쪽은 Bottleneck 통과 → [다시 concat] → cv2 → 출력
```

**split과 concat이 블록마다 최소 한 번씩 박혀 있다.** neck(FPN/PAN)도 여러 백본 단계의 feature를 concat으로 합치고, SPPF는 4-way concat을 쓴다. 분기가 예외적 상황이 아니라 **이 아키텍처의 기본 설계 패턴**이다.

실제로 미fusion된 90개의 block별 분포:

```
model.23: 18개   model.6: 9개   model.8: 9개   model.13: 9개
model.16: 9개    model.19: 9개  model.22: 6개  model.2: 4개
model.4: 4개     model.10: 4개
```

백본부터 neck, head까지 거의 전 구간에 골고루 퍼져 있다 — 특정 블록 하나가 문제가 아니라, **"conv 앞뒤로 분기가 없어야 한다"는 조건 자체가 CSP 기반 detector에서는 원래 만족하기 어려운 조건**이었다. 반대로 fusion에 성공한 12개(C2PSA attention 내부, Detect head 최종 1x1)가 오히려 예외적으로 분기 없는 구간이었던 것.

## 12. 분기돼도 INT8 연산 자체엔 문제없다 — 이건 ORT 구현의 사각지대다

**중요한 통찰**: 텐서가 여러 곳에서 읽힌다고 해서 그 값이 float로 돌아가야 하는 건 아니다. 데이터는 분기돼도 여전히 int8이다. 그런데 왜 ORT는 소비자가 여러 개면 U8 승격을 포기할까?

### 12.1 S8→U8 변환은 그냥 재표기(relabeling)다 — 수학적으로 항상 안전

```
real_value = scale × (q_s8 - zp_s8)
           = scale × ((q_s8+128) - (zp_s8+128))
           = scale × (q_u8 - zp_u8)          # q_u8 = q_s8+128, zp_u8 = zp_s8+128
```

값과 zero_point에 **똑같이** +128을 더하는 것뿐이라, 표현하는 실수값은 완전히 동일하다. 정보 손실 없는 순수한 재표기다. **소비자가 몇 개든 상관없이 항상 성립하는 등식.**

### 12.2 그런데 왜 ORT는 "소비자 1개"로 제한했나

`QDQ_S8_to_U8` 함수(소스에서 확인)는 이렇게 짜여 있다:

```cpp
Node& dq_node = *graph.GetNode(node.OutputNodesBegin()->Index());  // "첫 번째" 다음 노드 1개만 가져옴
```

Q 노드 하나 + 그 뒤에 붙은 **DQ 노드 딱 하나** 짝만 찾아서 같이 바꾼다. 소비자가 여러 개면 함수 진입 전에 `CheckOutputEdges(graph, node, 1)`로 아예 걸러버린다.

**이유는 구현의 단순함/안전성 때문으로 보인다:** 소비자가 여러 개인 경우까지 처리하려면 Q 노드 + 그 밑에 달린 DQ 노드 **전부**를 찾아서 **빠짐없이 동시에** 바꿔야 하는데, 이 "여러 개를 한 번에 일관되게 바꾸기" 로직은 구현·검증이 번거롭다. 그래서 "소비자 1개"라는 안전한 서브셋만 처리하도록 범위를 좁혀놓은 것 — **이론적 한계가 아니라 ORT 옵티마이저가 커버하지 않은 최적화 기회**다.

### 12.3 직접 하면 안전한가 — 조건은 "빠짐없이 전부"

소비자가 N개인 Q 노드를 U8로 승격하려면:

> **Q 노드 자체 + 그 밑에 달린 DQ 노드 N개 전부를 한 번에, 하나도 빠짐없이** zero_point +128, dtype S8→U8로 바꿔야 한다.

하나라도 놓치면: Q는 U8 값을 뱉는데 놓친 DQ 하나는 여전히 S8로 착각하고 해석 → **그 브랜치만 값이 조용히 128만큼 틀어져서 계산이 깨진다** (에러 없이 숫자만 틀림 — 가장 위험한 종류의 버그).

전부 일관되게 바꾸기만 하면 소비자 개수와 무관하게 안전하다.

### 12.4 실용적 구현 아이디어

ORT C++ 내부를 직접 고치는 것보다 현실적인 방법:

1. `InferenceSession`에 넘기기 **전에**, raw QDQ `.onnx` 파일을 파이썬(`onnx` 라이브러리)으로 직접 열기
2. 그래프 전체를 순회하며 S8인 Q/DQ 노드를 전부 찾기
3. **소비자 개수 상관없이 일괄적으로** zero_point +128, dtype S8→U8로 바꿔치기

이렇게 하면 ORT의 좁은 변환기가 손대기도 전에 그래프가 이미 전부 U8이라, `ConvNodeGroupSelector::Check`의 `dt_input==dt_output` 조건이 어디서든 저절로 만족되고 — 커널(U8 activation + S8 weight QLinearConv)은 fusion 성공했던 12개가 이미 실사용 중이므로 문제없을 것으로 보임 — **QOperator 수준의 fusion 비율을 QDQ 포맷을 유지한 채로도** 얻을 수 있을 가능성이 있다. (아직 실험은 안 해봄 — 다음 검증 후보.)

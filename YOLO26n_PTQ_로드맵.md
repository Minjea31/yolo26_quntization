# YOLO26n PTQ 실행 로드맵

> **대상**: 채널 프루닝(GM / L1 / L2)이 적용된 YOLO26n
> **배포 타겟**: RTX 3070 Laptop GPU (Ampere, sm86)
> **실험 환경**: RTX 3070 Laptop GPU 8GB (sm86) / CUDA 12.8 / Ubuntu 22.04 / conda `quant` (Python 3.11)
> **레포**: https://github.com/Minjea31/yolo26_pruning

---

## 환경 세팅 (완료 체크)

```bash
conda create -n quant python=3.11 -y
conda activate quant

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics onnx onnxsim
pip install tensorrt-cu12
pip install "nvidia-modelopt[onnx]"
```

검증:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
python -c "import tensorrt as trt; print(trt.__version__)"
```

- [ ] `get_device_capability()` → `(8, 6)` 확인
- [ ] TensorRT import 정상

**주의사항**

- PyTorch는 conda 채널 배포가 중단됨 → 반드시 pip 사용
- `pip install tensorrt`는 기본값이 CUDA 13 변종 → `tensorrt-cu12` 명시 필요
- pip 휠에는 `trtexec` CLI 미포함 (필요 시 tar/deb 별도 설치)

---

## Phase 0. 베이스라인 고정

> 가장 중요한 단계. 여기를 건너뛰면 이후 mAP 하락의 원인이 프루닝 탓인지 양자화 탓인지 분리 불가능해집니다.

### 체크리스트

- [ ] 체크포인트 목록 정리
  - [ ] 원본 YOLO26n FP32
  - [ ] GM(FPGM) 프루닝 버전 (비율별)
- [ ] 평가 파이프라인 고정
  - [ ] val 데이터셋 경로 및 버전 확정
  - [ ] mAP 측정 스크립트 단일화
  - [ ] 랜덤 시드 고정
  - [ ] 이미지 전처리(letterbox, 정규화) 파라미터 문서화
- [ ] 베이스라인 측정
  - [ ] 각 체크포인트 FP32 mAP50-95
  - [ ] 각 체크포인트 FP16 mAP50-95
  - [ ] 각 체크포인트 FP32 / FP16 latency (p50, p99)
- [ ] 결과를 `results.csv`에 기록

### 산출물

| 항목 | 내용 |
|---|---|
| `baseline_eval.py` | 고정된 평가 스크립트 |
| `results.csv` | 실험 ID, 설정, mAP, latency 누적 기록 |

---

## Phase 1. 통계 수집 & 민감도 가설 검증

### 체크리스트

- [ ] `register_forward_hook` 기반 통계 수집 스크립트 작성
- [ ] Calibration 이미지 100~500장으로 forward 수행
- [ ] Activation 통계 측정
  - [ ] min / max
  - [ ] percentile (99.9 / 99.99)
  - [ ] **kurtosis** (첨도)
  - [ ] std
  - [ ] outlier 비율
- [ ] Weight 통계 측정
  - [ ] per-channel min/max 분포
  - [ ] outlier 채널 존재 여부
- [ ] 구조적 가설과 실측 kurtosis 랭킹 대조

### 구조적 민감 레이어 가설 (검증 대상)

1. **Stem** — 입력 직후, 분포가 아직 정규화되지 않음
2. **C2PSA attention conv** — attention score의 dynamic range가 넓음
3. **C3k2 split-concat 접합부** — 서로 다른 분포의 텐서가 합쳐짐
4. **Detect head 출력 conv** — 회귀값의 스케일이 크고 정밀도 민감

### 핵심 주의사항

> **hook을 어디에 거는가가 결과를 좌우합니다.**
>
> YOLO는 Conv-BN-SiLU가 하나의 블록으로 묶여 있어서, Conv 출력 / BN 출력 / SiLU 출력의 분포가 완전히 다릅니다.
> TensorRT는 이 셋을 fuse하므로 **실제 양자화 대상은 SiLU 출력**입니다. 여기에 hook을 걸어야 합니다.

- BatchNorm이 분포를 정규화하기 때문에 min/max는 민감도 신호로 약합니다
- kurtosis와 percentile 기반 range가 더 유효한 지표입니다

---

## Phase 2. Fake Quantization 구현

### 체크리스트

- [ ] Quantize-Dequantize 함수 구현
  - [ ] scale / zero_point 계산
  - [ ] clamp
  - [ ] round (banker's rounding 여부 결정)
- [ ] Conv 모듈 wrapping 구조 설계 (**모듈 교체 방식 권장** — hook보다 제어가 쉬움)
- [ ] Calibration 방법 3종 구현
  - [ ] min-max
  - [ ] percentile (99.9 / 99.99)
  - [ ] KL-entropy (histogram 기반)
- [ ] 기본 설정: weight per-channel, activation per-tensor
- [ ] 전 레이어 INT8 fake quant → mAP 측정 (= 최악의 경우 기준선)

### 설계 메모

- Fake quantization은 **정확도 실험 전용 도구**입니다. 속도 측정과 혼동하면 불필요하게 복잡해집니다
- 실제 속도 검증은 Phase 7의 TensorRT에서만 수행
- modelopt 없이 순수 PyTorch로 직접 구현하는 것이 학습 목적에 부합

---

## Phase 3. Ablation 실험 (연구의 본체)

### 체크리스트

- [ ] **Calibration 방법 비교**
  - [ ] min-max
  - [ ] percentile 99.9
  - [ ] percentile 99.99
  - [ ] KL-entropy
- [ ] **Granularity 비교**
  - [ ] weight per-tensor vs per-channel
- [ ] **Calibration set 크기 영향**
  - [ ] 50장 / 100장 / 500장 / 1000장
- [ ] **레이어별 단독 민감도 측정 (핵심)**
  - [ ] 한 레이어만 INT8, 나머지 FP32 → 전 레이어 반복
  - [ ] mAP drop 기준 랭킹 생성
  - [ ] Phase 1의 kurtosis 가설과 일치 여부 판정
- [ ] **역방향 검증**
  - [ ] 한 레이어만 FP32, 나머지 INT8

### 기대 결과

Phase 1의 kurtosis 랭킹과 Phase 3의 실측 mAP drop 랭킹이 상관관계를 보이면,
값싼 통계량(kurtosis)으로 비싼 실험(레이어별 ablation)을 대체할 수 있다는 근거가 됩니다.
불일치한다면 그 자체가 흥미로운 발견입니다.

---

## Phase 4. Mixed-Precision 정책 확정

### 체크리스트

- [ ] Phase 3 랭킹 상위 N개 레이어만 FP16 유지
- [ ] N 스윕: 0, 2, 4, 8, 16
- [ ] "INT8 레이어 비율 vs mAP" 곡선 작성
- [ ] 곡선의 무릎(knee) 지점을 최종 정책으로 채택
- [ ] 정책을 JSON/YAML로 저장 (Phase 6에서 재사용)

### 산출물 형식 예시

```yaml
fp16_layers:
  - model.0.conv        # stem
  - model.10.attn.qkv   # C2PSA
  - model.23.cv3.2      # Detect head
int8_default: true
```

---

## Phase 5. ONNX Export

### 체크리스트

- [ ] 프루닝된 모델 FP32 export 성공 여부 먼저 확인
  - 채널 수가 불규칙해서 export가 깨질 수 있음
- [ ] **PyTorch 모듈명 ↔ ONNX 노드명 매핑 테이블 자동 생성 스크립트 작성**
- [ ] `onnxsim`으로 그래프 정리
- [ ] 수치 검증: ONNX Runtime FP32 출력 vs PyTorch 출력
  - [ ] max absolute difference 확인 (1e-4 이하 권장)
- [ ] Detect head / NMS를 그래프에 포함할지 결정
- [ ] opset 버전 결정 및 고정

### 핵심 주의사항

> PyTorch 모듈명(`model.10.cv1.conv`)과 ONNX 노드명(`/model.10/cv1/conv/Conv`)은 형식이 다릅니다.
> Phase 6에서 레이어별 precision을 지정하려면 이 매핑이 반드시 필요합니다.
> 수동으로 하면 실수하므로 스크립트로 자동화하세요.

---

## Phase 6. TensorRT 엔진 빌드

### 경로 선택

| 방식 | 설명 | 적합한 경우 |
|---|---|---|
| **Implicit** | `IInt8EntropyCalibrator2`로 calibration cache 생성 | 간단히 전체 INT8만 필요할 때 |
| **Explicit (QDQ)** | 그래프에 Q/DQ 노드 삽입 | **Phase 2~4에서 구한 scale을 그대로 이식할 때 (권장)** |

### 체크리스트

- [ ] Explicit QDQ 방식 채택 여부 결정
- [ ] Phase 4 정책대로 레이어별 precision 지정
  - [ ] `layer.precision = trt.float16`
  - [ ] `layer.set_output_type(0, trt.float16)`
  - [ ] `BuilderFlag.PREFER_PRECISION_CONSTRAINTS` 설정
- [ ] **verbose 빌드 로그 확인 (필수)**
  - [ ] 지정한 precision이 실제로 반영됐는지
  - [ ] layer fusion 때문에 무시되지 않았는지
- [ ] 엔진 직렬화 및 저장

### 핵심 주의사항

> **TensorRT는 precision 요청을 조용히 무시하는 경우가 많습니다.**
>
> fusion 과정에서 레이어가 병합되면 지정한 precision이 사라집니다.
> `PREFER_PRECISION_CONSTRAINTS`는 "선호"일 뿐이며, `OBEY_PRECISION_CONSTRAINTS`는 만족 불가 시 빌드 실패를 일으킵니다.
> verbose 로그를 열어서 최종 레이어별 precision을 눈으로 확인하는 과정을 생략하지 마세요.

---

## Phase 7. 속도 측정 & Roofline 검증

### 체크리스트

- [ ] Latency 측정 프로토콜 확립
  - [ ] warmup 충분히 (100회 이상)
  - [ ] 본 측정 1000회 반복
  - [ ] p50 / p99 기록
  - [ ] CUDA event 기반 타이밍 사용
- [ ] 조건별 비교
  - [ ] FP32
  - [ ] FP16
  - [ ] INT8 full
  - [ ] INT8 mixed (Phase 4 정책)
- [ ] **레이어별 프로파일링**
  - [ ] reformat 노드 위치 및 개수 파악
  - [ ] NCHW ↔ NC/32HW32 변환 비용 측정
- [ ] 프루닝 강도별 비교
  - [ ] 프루닝이 강할수록 INT8 이득이 줄어드는지 확인

### 예상 결과에 대한 사전 해석

> **3070 Laptop GPU에서 INT8 speedup이 거의 없게 나와도 그것은 실패가 아닙니다.**
>
> YOLO26n은 파라미터가 약 3M으로 작지만, 640px 입력과 multi-scale detection head 때문에 FLOPs는 상대적으로 높습니다.
> Roofline 관점에서 이 모델은 데스크탑 GPU에서 compute-bound 구간에 충분히 들어가지 못하고,
> INT8 변환으로 얻는 연산 이득보다 reformat 오버헤드가 더 클 수 있습니다.
>
> 또한 프루닝은 FLOPs를 제곱에 가깝게 줄이지만 reformat 비용은 선형으로만 줄어들기 때문에,
> 프루닝과 양자화는 부분적으로 경쟁 관계입니다.
>
> 이 예측이 실측으로 확인된다면, 그것 자체가 논지의 핵심 결과가 됩니다.

---

## 상시 관리 항목

- [ ] 모든 실험 결과를 단일 `results.csv`에 누적
  - 컬럼: `exp_id`, `checkpoint`, `prune_method`, `prune_ratio`, `quant_config`, `calib_method`, `calib_size`, `mAP50`, `mAP50-95`, `latency_p50`, `latency_p99`, `date`
- [ ] 스크립트는 `yolo26_pruning` 레포에 커밋
- [ ] Notion 페이지 갱신
  - **Quantization 개념 정리 (모델 무관)** — 일반화 가능한 발견
  - **YOLO26 전용 페이지** — 모델별 실측치와 구현 세부사항

---

## 전체 흐름 요약

```
Phase 0  베이스라인 고정
   ↓
Phase 1  통계 수집 (kurtosis) → 민감도 가설 수립
   ↓
Phase 2  Fake quantization 구현 → 최악 기준선 확보
   ↓
Phase 3  Ablation → 실측 민감도 랭킹 (가설 검증)
   ↓
Phase 4  Mixed-precision 정책 확정
   ↓
Phase 5  ONNX export + 노드명 매핑
   ↓
Phase 6  TensorRT 엔진 빌드 (레이어별 precision)
   ↓
Phase 7  속도 측정 → Roofline 예측 검증
```

**Phase 0~4는 정확도 트랙 (PyTorch fake quant), Phase 5~7은 속도 트랙 (TensorRT).**
두 트랙을 섞지 않는 것이 이 프로젝트의 기본 전략입니다.

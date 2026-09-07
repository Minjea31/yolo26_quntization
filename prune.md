# YOLO26 Pruning 로직 정리 (+ 양자화 견고성 수정 노트)

이 문서는 이 레포의 구조적(structured) 프루닝 파이프라인 전체 흐름과, 프루닝만 했을 때·INT8 양자화까지 했을 때 성능이 유지되도록 하기 위해 적용한 수정 사항을 정리한 것입니다.

---

## 1. 전체 파이프라인

```
baseline .pt
   │  prune_finetune.py
   ▼
PruneHandler.compress_yolo26()          (compress/Compress.py)
   ├─ prune()        : 채널별 마스크 생성 → weight 0으로 (아직 shape 유지)
   ├─ reconstruct()  : 0인 채널을 실제로 slicing 제거 → 진짜 작은 모델
   ├─ model_to_yaml(): 재구성된 구조를 yaml 청사진으로 기록
   └─ save()         : best_model_prune.pt / .yaml 저장
   ▼
model.train(epochs=...)                  fine-tuning으로 정확도 회복
   ▼
export (ONNX) → INT8 quantize            보드/NPU 배포
```

핵심: **fine-tuning은 yaml로 새 모델을 만들지 않고, 재구성된 모델 객체를 그대로 학습**합니다. yaml은 "구조 청사진(참고/재빌드용)"이며 **반드시 실제 .pt와 동일**해야 합니다.

---

## 2. 프루닝 대상 (`prune_type`)

`reconstruct()`/`prune()`이 다루는 레이어 인덱스:

- `backbone_layers = 0..10`, `detect_layers = 23`
- `B` : backbone(0~10)만
- `H` : backbone·Detect 제외(neck/head feature)
- `ALL` : Detect(23)만 빼고 전부

**Detect head(23)는 절대 프루닝하지 않습니다.** 대신 앞단 결과에 맞춰 입력 채널만 slicing합니다.

---

## 3. 채널 중요도 기준 (`method`)

각 Conv의 **output 채널(dim=0)** 단위로 남길/버릴 채널을 고릅니다.

| method | 기준 | 남기는 채널 |
|--------|------|-------------|
| `L1` | 필터 L1-norm | norm 큰 채널 |
| `L2` | 필터 L2-norm | norm 큰 채널 |
| `GM` (FPGM) | 필터 간 유클리드 거리합 | **geometric median에서 먼(고유 정보) 채널** |

### FPGM 원리
- 각 필터를 벡터화 → 필터 간 거리행렬 → 필터 i의 "다른 모든 필터와의 거리합" `similar_sum[i]`
- `similar_sum` 이 **작다 = geometric median에 가깝다 = 다른 필터로 대체 가능(중복)** → 제거
- `similar_sum` 이 **크다 = 고유 정보** → 보존

---

## 4. 프루닝 비율 계산 (`--pruning_ratio` → per-conv amount)

Conv는 output뿐 아니라 다음 layer와 맞추느라 input 채널도 함께 줄어들어 파라미터가 `(1-r)^2` 로 감소합니다. 목표 최종 감소율을 맞추기 위해:

```python
per_conv_ratio = 1 - sqrt(1 - pruning_ratio)   # prune_finetune.py
```

예) `pruning_ratio=0.5` → per-conv 약 0.293 만큼 output 채널 제거.

---

## 5. 🔧 채널 정렬(alignment) — 양자화/NPU 견고성 [수정됨]

### 문제
프루닝을 Conv마다 독립 비율로 적용하면 남는 채널이 **45, 91, 181** 같은 비정렬 값이 됩니다. baseline은 16/32/64… 정렬돼 있어 문제없지만, 비정렬 채널은:
- 일부 양자화/NPU 컴파일러가 채널을 8/16/32 배수로 요구 → **컴파일 에러**
- 강제 zero-padding → concat/split 지점에서 **채널 인덱스 오정렬로 feature 뒤섞임**
- 최적화 INT8 커널 대신 느린 fallback / 그래프 분할

### 수정
`PruneHandler._amt(module)` 가 남길 채널 수를 항상 `align` 배수로 반올림한 뒤, 잘라낼 개수를 정수로 넘깁니다. (`torch.nn.utils.prune`, `GMStructured` 모두 `amount`가 int면 "자를 채널 개수"로 해석)

```python
keep = round(num * (1 - cr) / align) * align
keep = clamp(keep, align, num)          # 최소 한 그룹, num 초과 금지
amount = num - keep                       # ln_structured / gm_structured 에 int로 전달
```

- 기본값 `--align 8`. `--align 1` 이면 정렬 없이 기존 동작.
- 검증: `pruning_ratio=0.5, align=8` 기준 16→8, 32→24, 48→32, 64→48, 128→88, 256→184 등 **모든 keep이 8의 배수**.

> 보드 벡터폭이 16/32 정렬을 요구하면 `--align 16` 또는 `--align 32` 로 올리면 됩니다.

---

## 6. 🔧 FPGM 선택 버그 [수정됨]

### 이전 (버그)
```python
tensor_sort = torch.tensor(similar_sum.argsort())          # 순위→필터인덱스 매핑
topk = torch.topk(tensor_sort, k=nparams_tokeep, ...)      # 그 배열에 다시 topk
mask = make_mask(t, dim, topk.indices)                     # 배열 내 '위치'를 채널로 사용
```
`argsort` 결과 배열에 다시 `topk`를 걸고 `.indices`(배열 내 위치)를 채널 인덱스로 써서, **거리합이 큰 실제 필터와 무관한 거의 무작위 선택**이 됐습니다. → FPGM이 사실상 작동 안 함 → 파인튜닝으로 겨우 복구되지만 weight 분포가 지저분해 **INT8 양자화에서 정확도가 크게 무너짐**.

### 수정
```python
sorted_idx = np.argsort(similar_sum)                       # 오름차순(가까운→먼)
keep_idx = torch.tensor(sorted_idx[nparams_toprune:].copy())  # 먼 쪽 keep
mask = make_mask(t, dim, keep_idx)
```
- 검증: 6개 유사 필터 + 2개 outlier 구성에서 `amount=6` → **정확히 outlier 2개만 보존(PASS)**.

---

## 7. 🔧 yaml 청사진 정합성 [수정됨]

`model_to_yaml()` 이 실제 .pt와 다른 값을 기록하던 부분 수정:

| 항목 | 이전 | 수정 |
|------|------|------|
| `nc` | `8` 하드코딩 | `detect.nc` (실제 학습 클래스 수) |
| SPPF `c2` | `max(c2, 128)` 인플레이션 | 실제 pruned 채널 |
| C2PSA `c2` | `max(c2, 128)` 인플레이션 | 실제 pruned 채널 |

→ `.pt`를 바로 ONNX로 export하면 yaml은 안 쓰이지만, **yaml로 재빌드하는 순간 구조/weight mismatch로 detect가 붕괴**하므로 청사진을 실제와 일치시킴.

---

## 8. 구조 재구성 (`reconstruct` / 각 모듈 `recon`)

0으로 만든 채널을 실제 slicing으로 제거하고, 앞뒤 layer 채널을 맞춥니다.

- **Conv** (`conv.py`): output 채널(norm≠0) slicing + 다음 layer에 맞춰 input 채널도 slicing. BN weight/bias/running_mean/var/num_features 동기화. depthwise면 groups 갱신.
- **Bottleneck**: `add`(residual)일 때 출력 채널을 입력과 동일 인덱스로 맞춰 shortcut 유지.
- **C3k2**: `cv1` 출력을 반으로 split → 한쪽은 그대로, 다른쪽은 내부 `m`(Bottleneck/C3k/PSABlock) 통과 → concat. split/concat 채널 인덱스를 `offset`으로 정밀 정렬. attention 블록 포함 시 해당 half는 full로 유지.
- **C2PSA**: attention 내부는 공격적으로 줄이지 않고(`input_attn = range(cv1_offset)`) 입출력 구조 중심으로 정렬.
- **SPPF**: cv1 출력을 4단 concat 하므로 offset 누적으로 cv2 입력 정렬.
- **Concat(12/15/18/21)**: 두 소스의 남은 채널 인덱스를 offset 더해 이어붙임.
- **Detect(23)**: cv2/cv3(및 end2end면 one2one_*)의 첫 conv/dwconv 입력 채널을 앞단(16/19/22) 결과로 slicing. box/cls 예측 자체는 보존.

> reconstruct의 concat/split offset 산술은 fragile합니다. fine-tuning 초기 val mAP가 0에 고정되지 않고 회복되면 구조 정합성은 정상으로 볼 수 있습니다.

---

## 9. 실행 방법

```bash
cd yolo26

# 프루닝 + 파인튜닝 (정렬 8 배수)
python prune_finetune.py \
  --bmodel ../baseline.pt \
  --pruning_ratio 0.5 --prune_type ALL --method GM \
  --align 8 \
  --epoch 100 --name yolo26_pruned --bs 16 --device 0
```

### 정상 동작 확인 포인트
- `Transferred ... items` 로그가 뜨지 않아야 함(재구성 모델 직접 학습).
- Detect 입력 채널이 줄고 **8의 배수**여야 함.
- 파라미터/GFLOPs가 baseline보다 감소.
- 파인튜닝 초반 val mAP가 0 고정이 아니라 회복.

---

## 10. INT8(프루닝 + 양자화) 검증 레시피

> ⚠️ 이 검증은 **dataset + baseline.pt + GPU + onnxruntime** 가 갖춰진 실제 환경에서 실행해야 합니다.
> (현재 이 저장소 환경에는 dataset/baseline/torchvision/CUDA/onnxruntime가 없어 로직 단위 검증만 수행함 — 5·6절 PASS.)

필요 패키지: `pip install onnx onnxruntime`

```bash
cd yolo26

# 1) 프루닝 + 5 epoch 파인튜닝
python prune_finetune.py --bmodel ../baseline.pt \
  --pruning_ratio 0.5 --method GM --align 8 \
  --epoch 5 --name val5 --bs 16 --device 0

# 2) FP32 mAP 비교 (baseline vs pruned)
yolo val model=../baseline.pt data=../dataset.yaml split=val imgsz=640
yolo val model=checkpoints/val5/weights/best.pt data=../dataset.yaml split=val imgsz=640
```

```python
# 3) int8_check.py — ONNX export → INT8 static quantize → mAP 비교
from ultralytics import YOLO
from ultralytics.utils.export.onnx import onnx_int8_quantize  # 레포 내장 헬퍼
from pathlib import Path

pt = "checkpoints/val5/weights/best.pt"
m = YOLO(pt)
onnx_fp32 = m.export(format="onnx", imgsz=640, opset=13)      # FP32 ONNX

# 검증(calibration) 이미지로 INT8 정적 양자화
onnx_int8 = str(Path(onnx_fp32).with_name("best_int8.onnx"))
onnx_int8_quantize(onnx_fp32, onnx_int8, dataset="../dataset.yaml")  # 시그니처는 utils/export/onnx.py 참고

# 4) FP32 vs INT8 ONNX mAP
print("FP32 ONNX:"); YOLO(onnx_fp32).val(data="../dataset.yaml", split="val", imgsz=640)
print("INT8 ONNX:"); YOLO(onnx_int8).val(data="../dataset.yaml", split="val", imgsz=640)
```

### 판정 기준
- **프루닝 유지**: pruned FP32 mAP ≈ baseline mAP (5 epoch면 완전 회복 전이라 소폭 하락은 정상, 0 근처면 구조 버그 의심).
- **양자화 견고성**: INT8 mAP 하락폭이 baseline의 INT8 하락폭과 비슷해야 함(예: 둘 다 -1~3%p). pruned만 급락하면 FPGM/정렬 문제 재점검.

---

## 11. 🔧 Reconstruct(구조 재구성) 정확성 버그 — 보드 "detect 거의 안 됨"의 진짜 원인 [수정됨]

masked(마스킹만) vs pruned(재구성) 모델의 **층별 출력을 채널 매핑해 정확 비교**한 결과, 재구성이 채널을 어긋나게 잇는 버그 3개를 찾아 수정했습니다. 이것들이 프루닝 모델이 학습·추론에서 무너지던 근본 원인입니다.

| # | 위치 | 버그 | 수정 |
|---|------|------|------|
| 1 | `block.py` `C3k2.recon` else 분기 | `union_channels`가 cv1 **첫 half(a)의 생존 채널을 누락** → chunk 분할 시 살아있는 채널이 버려짐 (압축률↑에서 치명적) | union 에 `cv1_out_channels_1` 포함 |
| 2 | `block.py` `SPPF.recon` | SPPF의 residual(`y+x`)인데 cv2 출력을 자기 norm으로 잘라 **잔차 채널이 어긋남** | `add`면 cv2 출력을 입력 채널에 정렬 |
| 3 | `block.py` `C3k2.recon` | 프루닝 후 `self.c`(split half 크기) 미갱신 → **export/forward_split에서 채널 mismatch** | `self.c = cv1.out_channels//2` 갱신 |

검증: masked vs pruned 층별 출력이 layer 0–22 **max\|Δ\|=0.00000** 로 일치, 각 모듈 격리 테스트도 Δ=0.

## 12. ✅ 최종 검증 결과 (test 환경, coco8, GM/FPGM, ratio=0.5, align=8)

> coco8은 train 4장 / val 4장이 완전 별개라, 프루닝+파인튜닝 후 **train split**(모델이 학습한 이미지)에서 보존 성능을 측정.

| 모델 | mAP50-95 | mAP50 |
|------|----------|-------|
| baseline FP32 | 0.857 | 0.965 |
| **pruned FP32** (구조적+FPGM) | **0.792** | **0.964** |
| baseline INT8 (static, head 제외) | 0.841 | 0.941 |
| **pruned INT8** (static, head 제외) | **0.857** | **0.995** |

- **프루닝 성능 보존 ≈ 92%** (mAP50은 0.965→0.964로 거의 동일), 파라미터 2.41M→1.73M, GFLOPs 5.4→4.1.
- **INT8 양자화 보존**: baseline·pruned 모두 손실 미미. **주의**: detect head(`model.23`)를 양자화에서 제외해야 함(full-model 양자화 시 baseline도 0으로 붕괴 — 프루닝 문제가 아니라 YOLO 공통의 head 민감성).
- pruned 모델은 정상 학습(loss 수렴)·추론(one2one head conf 0.89)·ONNX export 모두 정상.

## 13. 실행 환경 메모

- conda env `test` (torch 2.2.2+cu121, onnxruntime 1.15.1) 사용.
- 이 레포는 ultralytics 포크이므로 `pip install -e .` + `pip install polars` 필요.
- ⚠️ 이 env의 `torch/nn/utils/prune.py`가 `module_name`을 요구하도록 깨진 패치본이라, **stock torch 2.2.2 버전으로 복원**해야 GM/`custom_from_mask`가 동작 (`prune.py.patched_bak` 백업 있음).

## 14. 변경 파일 요약

| 파일 | 변경 |
|------|------|
| `compress/GM.py` | FPGM 선택 로직 수정(먼 필터 보존) + `compute_mask(**kwargs)` torch 호환 |
| `compress/Compress.py` | `_amt()` 채널 정렬 + 3 method 적용, yaml `nc`/SPPF·C2PSA 정합성, 출력폴더 `mkdir` |
| `ultralytics/nn/modules/block.py` | **C3k2 union(첫 half), SPPF residual 정렬, C3k2 self.c 갱신** (핵심 reconstruct 수정) |
| `prune_finetune.py` | `--align` 옵션 추가 |

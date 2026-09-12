# YOLO26 Pruning & Quantization

Ultralytics YOLO26 fork에 구조적(channel) pruning을 적용하고, pruned 모델을 fine-tuning한 뒤 배포용으로 양자화까지 검토한 프로젝트입니다. 크게 **원본(Baseline) 학습 → Pruning → Quantization** 세 단계로 구성됩니다.

## 프로젝트 구조

```
.
├── asset/                 # 구조 다이어그램, 환경 복구용 참고 파일
│   ├── C2PSA.jpeg         # C2PSA 블록 구조 및 channel 변수 정리 이미지
│   └── prune.py           # torch.nn.utils.prune 스톡(stock) 버전 백업 (아래 "환경 메모" 참고)
├── dataset.yaml           # 학습/검증/테스트 데이터 경로 및 클래스 설정
├── quantization/          # PTQ(양자화) 민감도 분석 스크립트 및 결과
│   ├── kurtosis_check.py  # 레이어별 activation kurtosis/clipping 분석 (양자화 민감 레이어 탐색)
│   ├── kernel_check.py    # INT8 ONNX가 ORT에서 실제로 어떤 커널로 fusion/실행되는지 확인 (graph dump + profiling)
│   ├── kernel_diff.py     # kernel_check.py 결과에서 fusion된 Conv와 안 된 Conv를 대조 분석
│   ├── exclude_lists.json # 양자화 제외 레이어 목록 (DETECT_HEAD / TIER1~3)
│   └── calib_actstats.npy # kurtosis_check.py 실행 결과 캐시 (git 미추적, 재생성 가능)
├── model_save.py          # 모델 구조를 텍스트로 덤프하는 디버그 스크립트
├── model_structure_pdb.py # 모델 구조 확인용 pdb 디버그 스크립트
├── prune.md               # Pruning 파이프라인/버그 수정 상세 기록
└── yolo26/                 # Ultralytics fork 본체 (원본 학습 + pruning 로직 포함)
    ├── train_baseline_model.py  # 원본(baseline) 모델 학습
    ├── prune_finetune.py        # pruning + fine-tuning 실행
    ├── eval.py                  # 모델 평가(mAP)
    └── compress/
        ├── Compress.py    # pruning, 구조 재구성(reconstruct), checkpoint/yaml 저장
        └── GM.py           # GM(Geometric Median, FPGM) 기반 structured pruning
```

`dataset/`, `model/`, `runs/`, 각종 가중치(`*.pt`, `*.onnx`, `*.engine` 등)는 용량이 크고 재생성 가능해 `.gitignore`로 제외되어 있습니다. 로컬에는 그대로 두고 작업하면 됩니다.

## 설치

```bash
cd yolo26
conda create -n yolo26 python=3.10 -y
conda activate yolo26
pip install -r requirements.txt
pip install polars
pip install -e .
```

## 데이터셋

리포지토리 루트의 `dataset/` 디렉터리에 `train/val/test` 이미지·라벨을 배치합니다. `dataset.yaml`은 이를 기준으로 경로를 잡고, 현재 클래스는 `nc: 1` (`CAR`)입니다.

```yaml
train: dataset/train/images
val: dataset/val/images
test: dataset/test/images
```

---

## 1. 원본(Baseline) 모델

`yolo26/train_baseline_model.py`로 pruning 없이 baseline YOLO26n을 학습합니다. pruning 이후 성능 비교의 기준점이 됩니다.

```bash
cd yolo26
python train_baseline_model.py \
  --model_pt ../yolo26n.pt \
  --data ../dataset.yaml \
  --name baseline \
  --bs 4 \
  --epoch 100 \
  --device 0
```

학습 결과는 `yolo26/checkpoints/<name>/`에 저장됩니다. 중단 후 이어서 학습하려면 `--resume` 플래그를 추가합니다.

---

## 2. Pruning

`yolo26/prune_finetune.py`가 pruning + fine-tuning을 한 번에 수행합니다. 내부 로직(채널 정렬, FPGM 선택 로직, reconstruct 시 concat/split offset 처리 등)의 상세 배경과 버그 수정 이력은 [`prune.md`](prune.md)에 정리되어 있습니다.

```bash
cd yolo26
python prune_finetune.py \
  --bmodel ../baseline.pt \
  --pruning_ratio 0.5 \
  --prune_type ALL \
  --method GM \
  --align 8 \
  --cfg_output_path prune \
  --epoch 100 \
  --name yolo26_pruned \
  --bs 16 \
  --device 0
```

### 주요 옵션

- `--bmodel`: pruning할 baseline/pretrained checkpoint 경로
- `--pruning_ratio`: 최종 파라미터/FLOPs 감소 목표 비율 (예: `0.5` → 약 50%)
- `--prune_type`: `B`(backbone) / `H`(neck·head) / `ALL`(Detect head 제외 전부)
- `--method`: 채널 중요도 기준 — `L1`, `L2`, `GM`(FPGM, 권장)
- `--align`: 남길 채널 수를 맞출 배수 (기본 `8`). 양자화/NPU 배포 시 비정렬 채널로 인한 컴파일 실패·채널 뒤섞임을 방지
- `--cfg_output_path`: pruned checkpoint/yaml 저장 디렉터리
- `--epoch`, `--name`, `--bs`, `--device`: fine-tuning 설정
- `--resume_path`: 저장된 checkpoint에서 resume할 때 사용

`--pruning_ratio`는 최종 감소율 기준이며, 내부적으로 Conv output pruning amount는 `amount = 1 - sqrt(1 - pruning_ratio)`로 환산됩니다(input/output 채널이 함께 줄어드는 효과 보정).

### 출력 산물 (`--cfg_output_path prune` 기준)

- `yolo26/prune/best_model_prune.pt`: 구조 재구성까지 완료된 pruned checkpoint
- `yolo26/prune/best_model_prune.yaml`: pruned 구조 참고용 yaml (fine-tuning에는 사용되지 않고 확인용)
- `yolo26/checkpoints/<name>/`: fine-tuning 결과

### 검증 결과 (coco8, GM/FPGM, ratio=0.5, align=8)

| 모델 | mAP50-95 | mAP50 |
|------|----------|-------|
| baseline FP32 | 0.857 | 0.965 |
| **pruned FP32** | **0.792** | **0.964** |

파라미터 2.41M → 1.73M, GFLOPs 5.4 → 4.1, 성능 보존 약 92%. 자세한 검증 절차·판정 기준은 [`prune.md`](prune.md) 참고.

---

## 3. Quantization (PTQ)

배포 가속을 위해 INT8 PTQ(Post-Training Quantization)와 TensorRT FP16을 비교 검증했습니다.

- `quantization/kurtosis_check.py`: baseline/pruned 모델의 Conv 레이어별 activation kurtosis·percentile range·clipping 여부를 측정해 양자화에 취약한 레이어를 탐색합니다. `../model/best.pt`, `../dataset/train/images`를 기준으로 실행하므로 리포지토리 루트 기준 `cd quantization && python kurtosis_check.py`로 실행합니다.
- `quantization/exclude_lists.json`: 위 분석과 실측을 바탕으로 양자화에서 제외할 ONNX 노드 목록(`DETECT_HEAD`, `TIER1~3` 민감도 등급)을 정리한 참고 데이터입니다.

### 결론 (2026-09 기준)

- **Detect head(`model.23`)는 반드시 양자화에서 제외**해야 합니다. 포함 시 baseline·pruned 모두 mAP가 0으로 붕괴합니다(YOLO 공통 이슈로, pruning과 무관).
- Detect head를 제외한 INT8 PTQ 자체의 정확도 손실은 크지 않았지만(FP32 ONNX val mAP50-95 0.828 → INT8 0.801~0.819), **RTX 3070 Laptop GPU 기준 실측 속도에서는 INT8이 TensorRT FP16보다 오히려 느렸습니다.** 이 모델(YOLO26n + pruning)은 이미 연산 병목에서 멀어져 있어 INT8이 절약하는 연산 시간보다 INT8↔FP16 reformat 비용이 더 큰 것이 원인입니다.
- 최종적으로 **TensorRT FP16을 배포 포맷으로 채택**했습니다. INT8 방향은 이 리포지토리 범위에서는 종료된 상태입니다.
- 전체 실행 로그와 실측 수치, 실험 배경은 Notion 문서에 정리되어 있습니다: [yolo26_pruning_quantization](https://app.notion.com/p/3cee39b74831812d82e6f12b6feef6b2?source=copy_link)

### INT8 커널 선택 분석 (왜 INT8이 안 빨랐는가)

ONNX Runtime의 `quantize_static`은 기본적으로 **QDQ 포맷**(`Conv` 앞뒤에 `QuantizeLinear`/`DequantizeLinear`만 삽입)으로 양자화하며, 이 자체로는 아직 INT8 커널이 아닙니다. 실제 커널 선택은 ORT가 세션을 로드할 때 그래프 최적화 단계에서 QDQ 패턴을 네이티브 INT8 커널(`QLinearConv` 등)로 fusion할 수 있는지 판단해서 결정됩니다. `quantization/kernel_check.py`, `quantization/kernel_diff.py`로 이걸 직접 측정했습니다.

**측정 방법**: `baseline.pt` → ONNX export → `quantize_static`으로 INT8 양자화(`model/baseline_int8.onnx`) → `CPUExecutionProvider`로 세션 생성 시 `optimized_model_filepath`로 최적화된 그래프를 덤프 + 프로파일링.

**결과 (CPU, YOLO26n baseline, Conv 102개 기준)**:

| 구분 | 개수 |
|---|---|
| `QLinearConv`(등)로 fusion된 Conv | **12개** |
| `DequantizeLinear → Conv(FP32) → QuantizeLinear`로 남은 Conv | **90개** |

즉 대부분의 Conv가 진짜 INT8 커널을 타지 못하고, "INT8 텐서를 FP32로 풀었다가 연산 후 다시 INT8로 양자화"하는 캐스팅 경로로 실행됩니다. 프로파일링 실측 누적 시간도 이를 뒷받침합니다.

| op | 누적 시간(us) |
|---|---|
| `DequantizeLinear` | 20,459 |
| `Conv`(FP32 폴백) | 14,797 |
| `QuantizeLinear` | 6,967 |
| `QLinearConv`(진짜 INT8) | 273 |

**`DequantizeLinear` + `QuantizeLinear` 캐스팅 오버헤드(27,426us)가 실제 Conv 연산 시간(14,797us)보다 크고, 진짜 INT8 커널(`QLinearConv`, 273us)이 쓰는 시간은 무시할 수준입니다.** 이것이 위 결론에서 "INT8이 TensorRT FP16보다 오히려 느렸다"는 현상의 CPU/ORT 레벨에서의 구체적 메커니즘입니다 — INT8 연산 자체의 이득보다 양자화·역양자화 캐스팅 비용이 더 크기 때문입니다.

`kernel_diff.py`로 어떤 Conv가 fusion됐는지 위치를 확인해보면, fusion된 12개는 **C2PSA attention 블록(`model.10`, `model.22`의 `attn.pe`/`attn.proj`/`ffn.1`)과 end2end Detect head의 최종 1x1 conv(`model.23`의 `one2one_cv2.*.2`/`one2one_cv3.*.2`)에 집중**돼 있고, 백본 대부분의 Conv(`model.2`~`model.19`)는 전부 미fusion 상태입니다.

**왜 12개만 fusion됐는지 원인을 ONNX Runtime 1.18.0 소스로 확정**했습니다. `ConvNodeGroupSelector::Check()`(`qdq_selectors.cc`) 자체의 조건(activation/weight dtype, INT32 bias, `bias_scale==act_scale*w_scale` 정확한 일치, `group`/`kernel_shape`/`strides`/`pads`)은 fused·unfused 102개 전부 동일하게 통과해서 원인이 아니었고, 진짜 원인은 그 앞 단계인 **`QDQS8ToU8Transformer`**(`onnxruntime/core/optimizer/qdq_transformer/qdq_s8_to_u8.cc`)에 있었습니다.

- `quantize_static`은 기본값으로 **부호 있는 INT8(S8)**로 양자화하는데, CPU의 빠른 INT8 커널은 **부호 없는 UINT8(U8)** 기준이라 ORT가 QLinearConv fusion 전에 S8→U8 승격을 먼저 시도합니다.
- 이 승격은 `QDQ::MatchQNode(node) && optimizer_utils::CheckOutputEdges(graph, node, 1)` 조건, 즉 **해당 텐서를 만드는 Q 노드의 소비자(consumer)가 정확히 1개일 때만** 적용됩니다.
- 실측 결과 `baseline_int8.onnx`의 DequantizeLinear 124개가 **소비자 2개 이상(shared)** 이었습니다 — YOLO26의 C3k2/SPPF/concat 구조상 backbone·neck 대부분의 conv 출력이 다음 레이어 + 스킵 커넥션/concat 등 여러 곳에서 재사용되기 때문입니다. 예: `/model.0/conv/Conv`는 입력은 UINT8로 변환됐지만 출력은 fan-out=2라 변환이 안 돼 INT8로 남음 → `ConvNodeGroupSelector::Check()`의 `dt_input != dt_output` 조건에 걸려 fusion 거부.
- 반면 C2PSA attention 블록(`attn.pe→attn.proj→ffn.1`)과 Detect head의 최종 `one2one_cv2/cv3` 1x1 conv는 **분기 없는 순수 직선 체인**이라 입출력 양쪽 다 UINT8로 깨끗이 변환되어 fusion에 성공했습니다.

즉 **"이 모델이 branching이 많은 구조(C3k2 split/concat, SPPF, skip connection)라서 ORT의 S8→U8 내부 최적화가 대부분 노드에서 막히고, 그 결과 QLinearConv 대신 캐스팅 경로로 빠진다"**가 유력한 설명입니다.

✅ **gdb 라이브 디버깅으로 최종 확정.** `/model.0/conv/Conv`의 출력 Q 노드를 raw ONNX 파일(quantize_static 직후)에서 정적으로 세어보면 소비자(consumer)가 1개뿐이라 `QDQS8ToU8Transformer`의 `CheckOutputEdges(graph, node, 1)` 조건을 통과해야 정상인데, 실제로는 변환되지 않았습니다. ONNX Runtime을 `--config Debug`로 직접 빌드해 `qdq_s8_to_u8.cc:95`(`QDQS8ToU8Transformer::ApplyImpl`의 `if (QDQ::MatchQNode(node) && CheckOutputEdges(...))`)에 브레이크포인트를 걸고 해당 노드에서 멈춰 직접 확인한 결과:

```
(gdb) print optimizer_utils::CheckOutputEdges(graph, node, 1)
$3 = false
(gdb) print node.GetOutputEdgesCount()
$5 = 2
```

**raw 파일에서는 소비자가 1개였던 Q 노드가, `QDQS8ToU8Transformer`(Level 2)가 실행되는 시점에는 실제로 소비자 2개를 가지고 있었습니다.** 즉 ORT가 세션을 로드하며 먼저 실행하는 Level 1 최적화 패스들 중 하나가, quantize_static이 만든 원본 그래프에는 없던 두 번째 소비자 엣지를 이 텐서에 추가한 것입니다. 이건 파이썬으로 ONNX 파일을 아무리 정적으로 뜯어봐도 볼 수 없고, 실행 중인 프로세스를 gdb로 직접 들여다봐야만 확인 가능한 부분이었습니다 — 정적 소스/그래프 분석의 한계를 실제로 넘어서서 근본 원인을 확정한 사례입니다.

### CPU vs CUDAExecutionProvider 비교 — GPU는 아예 실행되지 않았다

같은 `baseline_int8.onnx`를 `kernel_check.py --provider CUDAExecutionProvider`로 돌려봤더니, 세션 초기화 로그에 다음과 같이 찍혔습니다:

```
All nodes placed on [CPUExecutionProvider]. Number of nodes: 1391
```

`CUDAExecutionProvider`를 요청했는데도 **1391개 노드 전부가 CPU로 배정**됐고, 실행 시간도 CPU 단독 실행과 오차 범위 내로 동일했습니다(`DequantizeLinear` 20,459→20,808us, `Conv` 14,797→14,647us 등). ONNX Runtime은 요청한 provider 목록에 항상 `CPUExecutionProvider`를 fallback으로 추가하는데, 이 그래프의 QDQ/`QLinear*` 연산자 조합에 대해 CUDA EP가 커널을 갖고 있지 않아 그래프 파티셔너가 전체를 CPU로 떨어뜨린 것으로 보입니다.

**결론: `quantize_static`(QDQ)로 만든 INT8 ONNX는 순수 `CUDAExecutionProvider`로는 GPU 가속을 전혀 받지 못합니다.** GPU에서 진짜 INT8 가속을 받으려면 완전히 다른 경로(TensorRT execution provider 또는 TensorRT 엔진 직접 export)가 필요하며, 이는 위 "최종적으로 TensorRT FP16을 배포 포맷으로 채택" 결론이 왜 타당했는지를 실측으로 뒷받침합니다.

---

## 환경 메모

- `asset/prune.py`는 `torch.nn.utils.prune`의 스톡(stock) 버전 백업입니다. 일부 conda 환경의 `torch/nn/utils/prune.py`가 `module_name`을 요구하도록 깨진 패치본으로 설치되어 있으면 GM/`custom_from_mask`가 동작하지 않으므로, 해당 파일로 교체해 복원합니다.
- pruning 정상 동작 확인 포인트: 학습 시작 시 `Transferred ... items from pretrained weights` 로그가 뜨지 않아야 하고, Detect 입력 채널이 줄어 있어야 하며(예: `[45, 91, 181]`), 파인튜닝 초반 val mAP가 0에 고정되지 않고 회복되어야 합니다.

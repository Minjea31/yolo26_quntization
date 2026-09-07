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
- Detect head를 제외한 INT8 PTQ 자체의 정확도 손실은 크지 않았지만(FP32 ONNX val mAP50-95 0.828 → INT8 0.801~0.819), **RTX 3050 Ti 기준 실측 속도에서는 INT8이 TensorRT FP16보다 오히려 느렸습니다.** 이 모델(YOLO26n + pruning)은 이미 연산 병목에서 멀어져 있어 INT8이 절약하는 연산 시간보다 INT8↔FP16 reformat 비용이 더 큰 것이 원인입니다.
- 최종적으로 **TensorRT FP16을 배포 포맷으로 채택**했습니다. INT8 방향은 이 리포지토리 범위에서는 종료된 상태입니다.
- 전체 실행 로그와 실측 수치, 실험 배경은 Notion 문서에 정리되어 있습니다: [yolo26_pruning_quantization](https://app.notion.com/p/3cee39b74831812d82e6f12b6feef6b2?source=copy_link)

---

## 환경 메모

- `asset/prune.py`는 `torch.nn.utils.prune`의 스톡(stock) 버전 백업입니다. 일부 conda 환경의 `torch/nn/utils/prune.py`가 `module_name`을 요구하도록 깨진 패치본으로 설치되어 있으면 GM/`custom_from_mask`가 동작하지 않으므로, 해당 파일로 교체해 복원합니다.
- pruning 정상 동작 확인 포인트: 학습 시작 시 `Transferred ... items from pretrained weights` 로그가 뜨지 않아야 하고, Detect 입력 채널이 줄어 있어야 하며(예: `[45, 91, 181]`), 파인튜닝 초반 val mAP가 0에 고정되지 않고 회복되어야 합니다.

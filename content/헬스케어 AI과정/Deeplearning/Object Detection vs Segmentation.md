---
title: 054 Object Detection vs Segmentation
tag:
  - 헬스케어 ai
  - deeplearing
  - tensorflow
description: 260713 수업 내용 정리
---

# Object Detection vs Segmentation

컴퓨터 비전 분야에서 이미지를 "이해"하는 수준은 단계적으로 깊어진다. 단순히 "이 이미지에 고양이가 있다"는 것을 아는 것에서 시작해서, 어디에 있는지 경계를 픽셀 단위로 구분하는 것까지 다양한 수준의 태스크가 존재한다. 의료 영상 분야에서는 이 기술이 종양 위치 파악, 장기 경계 추출, 내시경 이미지 분석 등에 핵심적으로 활용된다.

---

## 1. 컴퓨터 비전 태스크의 4단계

이미지를 이해하는 수준에 따라 네 가지 태스크로 나뉜다.

```
수준 낮음 ──────────────────────────────── 수준 높음

Classification     Classification    Object         Instance
                   + Localization    Detection      Segmentation
     ↓                   ↓               ↓               ↓
"고양이다"         "고양이 위치는   "개 3마리,       "각 개체를
                   여기다"          고양이 1마리,    픽셀 단위로
                                    각각 이 위치"    분리해서 표시"
No objects,        Single Object    Multiple        Multiple
just pixels                         Objects         Objects
```

### Classification (분류)

이미지 전체를 보고 어떤 클래스인지 하나의 레이블을 출력한다. "이 이미지는 고양이다." 위치 정보는 없다.

### Classification + Localization (분류 + 위치)

하나의 객체를 분류하고 **Bounding Box**(경계 상자)로 위치를 표시한다. 객체가 하나인 경우에 적합하다.

### Object Detection (객체 탐지)

**여러 객체**를 동시에 찾아서 각각에 Bounding Box와 클래스 레이블을 부여한다. "개 3마리가 이 좌표에, 고양이 1마리가 저 좌표에 있다"처럼 복수 객체를 처리한다.

```
출력 형식:
  [(클래스: "DOG",  신뢰도: 0.95, bbox: [x1,y1,x2,y2]),
   (클래스: "DOG",  신뢰도: 0.88, bbox: [x1,y1,x2,y2]),
   (클래스: "CAT",  신뢰도: 0.92, bbox: [x1,y1,x2,y2])]
```

### Semantic Segmentation (의미론적 분할)

픽셀 하나하나에 클래스 레이블을 부여한다. 단, 같은 클래스의 여러 개체를 구분하지 않는다. "이 픽셀들은 잔디, 저 픽셀들은 고양이, 저쪽은 하늘"처럼 픽셀 단위로 분류한다.

```
예시: 풀밭에 고양이 두 마리
  Semantic Segmentation: 모든 고양이 픽셀 = 같은 색 (개체 구분 없음)
  Instance Segmentation: 고양이1 픽셀 = 빨강, 고양이2 픽셀 = 파랑 (개체 구분)
```

### Instance Segmentation (인스턴스 분할)

Semantic Segmentation + Object Detection의 결합이다. 같은 클래스라도 **개체별로 픽셀을 구분**한다. 의료 영상에서 세포 각각을 독립적으로 구분하거나, 도로 위 차량 각각의 경계를 추출하는 데 사용한다.

---

## 2. 평가 지표 — IOU와 Dice Coefficient

### IOU (Intersection Over Union)

예측 영역과 실제 영역(Ground Truth)이 얼마나 겹치는지를 측정한다. Object Detection과 Segmentation 모두에서 가장 기본적인 평가 지표다.

$$IOU = \frac{\text{Area of Overlap (겹치는 면적)}}{\text{Area of Union (합집합 면적)}}$$

```
예측 박스:    ┌──────┐
              │      │
Ground Truth: │  ┌───┼──┐
              └──┼───┘  │
                 └───────┘

Area of Overlap: 두 박스가 겹치는 부분
Area of Union:   두 박스를 합친 전체 부분

IOU = 0.0: 전혀 겹치지 않음 (완전히 틀린 예측)
IOU = 0.5: 50% 겹침 (기준값으로 많이 사용)
IOU = 1.0: 완벽하게 일치
```

일반적으로 IOU ≥ 0.5를 True Positive(올바른 탐지)로 판정하고, 미만이면 False Positive(잘못된 탐지)로 처리한다.

### Dice Coefficient (다이스 계수)

의료 영상 Segmentation 평가에 특히 많이 사용되는 지표다. IOU와 개념은 비슷하지만 **겹치는 영역에 2를 곱해서 가중치**를 더 주는 방식이다.

$$\text{Dice} = \frac{2 \times \text{Area of Overlap}}{\text{총 예측 면적} + \text{총 실제 면적}}$$

```
직관:
  IOU   = overlap / (A + B - overlap)
  Dice  = 2 × overlap / (A + B)

  → Dice는 IOU보다 겹치는 영역에 더 민감하게 반응
  → 의료 영상처럼 작은 영역(종양)도 정확하게 잡아야 할 때 유용

값 범위: 0 ~ 1
  0: 전혀 일치하지 않음
  1: 완벽하게 일치
```

Dice와 IOU의 관계: $$\text{Dice} = \frac{2 \times \text{IOU}}{1 + \text{IOU}}$$

---

## 3. U-Net — 의료 영상 Segmentation 표준 구조

U-Net은 2015년 Freiburg 대학에서 의료 영상 Segmentation을 위해 개발된 CNN 구조다. 적은 데이터로도 높은 성능을 내는 구조 덕분에 의료 분야 Segmentation의 표준이 되었다.

### U-Net의 핵심 아이디어

일반 CNN은 층을 거치면서 공간 해상도가 점점 줄어든다. 이것은 분류 문제에선 괜찮지만, Segmentation처럼 **픽셀 단위로 출력이 필요한 경우** 해상도를 다시 복원해야 한다. U-Net은 두 가지 경로를 결합해서 이 문제를 해결한다.

```
U-Net 전체 구조 ("U" 모양):

입력 이미지
    ↓
┌─────────────────────────────────────────────┐
│           Contracting Path (인코더)           │
│  (해상도 줄이며 특성 추출, max pooling)       │
│                                              │
│  [128×128] → [64×64] → [32×32] → [16×16]   │
│     ↓           ↓         ↓         ↓        │
└──── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ Skip ────┘
                                    Connection
┌─────────────────────────────────────────────┐
│           Expansive Path (디코더)             │
│  (해상도 복원, up-conv 2×2)                  │
│                                              │
│  [16×16] → [32×32] → [64×64] → [128×128]   │
└─────────────────────────────────────────────┘
    ↓
Segmentation Map (픽셀별 클래스)
```

### Skip Connection — U-Net의 핵심

Contracting Path에서 추출한 특성맵을 Expansive Path의 대응되는 층에 직접 연결(copy and crop)한다. 이를 통해:

```
Contracting Path가 학습하는 것: "무엇이 있는가" (고수준 의미 정보)
Expansive Path가 복원해야 하는 것: "정확히 어디에 있는가" (공간 정보)

문제: 해상도를 줄이는 과정에서 정확한 위치 정보가 손실됨
해결: Skip Connection으로 인코더의 공간 정보를 디코더에 직접 전달

→ "의미는 알고 있고, 정확한 위치도 알고 있다" = 정밀한 Segmentation
```

### 구성 요소

```
Contracting Path (인코더):
  - 3×3 conv + ReLU (특성 추출)
  - 3×3 conv + ReLU (특성 추출)
  - 2×2 max pool (해상도 절반으로 축소)
  × N회 반복

Expansive Path (디코더):
  - 2×2 up-conv (해상도 2배로 복원)
  - Skip connection concat (인코더의 특성맵과 연결)
  - 3×3 conv + ReLU × 2

최종 출력:
  - 1×1 conv (채널 수 → 클래스 수)
  - Segmentation Map
```

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(x, filters):
    """U-Net의 기본 블록: Conv → ReLU → Conv → ReLU"""
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return x

def build_unet(input_shape=(512, 512, 1), num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)

    # ── Contracting Path (인코더) ─────────────────────
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D(2)(c1)         # 512→256

    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D(2)(c2)         # 256→128

    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D(2)(c3)         # 128→64

    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D(2)(c4)         # 64→32

    # ── Bottleneck ────────────────────────────────────
    c5 = conv_block(p4, 1024)

    # ── Expansive Path (디코더) ───────────────────────
    u6 = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
    u6 = layers.Concatenate()([u6, c4])     # Skip connection
    c6 = conv_block(u6, 512)

    u7 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
    u7 = layers.Concatenate()([u7, c3])
    c7 = conv_block(u7, 256)

    u8 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
    u8 = layers.Concatenate()([u8, c2])
    c8 = conv_block(u8, 128)

    u9 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
    u9 = layers.Concatenate()([u9, c1])
    c9 = conv_block(u9, 64)

    # ── 출력 ─────────────────────────────────────────
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(c9)

    return Model(inputs, outputs, name='U-Net')

model = build_unet(input_shape=(512, 512, 1), num_classes=2)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.MeanIoU(num_classes=2)]
)
model.summary()
```

---

## 4. U-Net의 발전형 — nnU-Net, UnetR, Swin UNet

### nnU-Net (No New U-Net)

"새로운 것 없는 U-Net"이라는 이름이 역설적이다. 2021년 독일 DKFZ에서 발표했으며, U-Net 구조 자체보다 **데이터와 학습 설정을 자동으로 최적화**하는 파이프라인이 핵심이다.

```
nnU-Net의 자동화 항목:
  - 입력 이미지 크기에 따른 패치 크기 자동 결정
  - 배치 크기 자동 결정 (GPU 메모리 기반)
  - 정규화 방법 자동 선택 (Z-score / CT HU 기반 / 없음)
  - 2D / 3D / 3D cascade 모델 자동 선택
  - 앙상블 여부 자동 결정

결과:
  의료 영상 Segmentation 10개 이상 벤치마크에서 최고 성능
  "튜닝 없이도 전문가 수준" → 실무에서 baseline으로 자주 사용
```

### UnetR (U-Net Transformers)

Transformer의 Encoder를 U-Net의 인코더로 대체한 구조다. 3D 의료 영상(CT, MRI)에 특화되어 있으며, Vision Transformer(ViT)가 이미지를 패치 단위로 처리하는 방식을 U-Net 디코더와 결합한다.

```
기존 U-Net:   CNN 인코더 + CNN 디코더
UnetR:        Transformer 인코더 + CNN 디코더

장점:
  CNN은 지역적 패턴(엣지, 질감) 잘 포착
  Transformer는 전역적 문맥(장기 전체 구조) 잘 포착
  → 두 장점을 결합
```

### Swin UNet (Shifted Window U-Net)

Swin Transformer를 U-Net에 결합한 구조다. Swin Transformer는 이미지를 겹치지 않는 작은 윈도우로 나눠서 각 윈도우 내에서 Attention을 계산하고, 윈도우를 이동(Shifted)해서 윈도우 간 정보도 교환한다.

```
일반 ViT:    전체 이미지의 모든 패치 간 Attention → 계산량 O(n²)
Swin ViT:   작은 윈도우 내 Attention + 윈도우 이동 → 계산량 O(n)

장점: 고해상도 의료 영상(512×512, 1024×1024)에서도 효율적
```

---

## 5. 의료 영상 데이터 형식 — DICOM vs NIfTI

의료 영상은 일반 사진 파일(.jpg, .png)과 다른 특수한 형식을 사용한다.

### DICOM (Digital Imaging and Communications in Medicine)

병원에서 사용하는 의료 영상 표준 형식이다. PACS(Picture Archiving and Communication System) 서버에 저장되고 관리된다.

```
DICOM 특징:
  - 하나의 이미지가 하나의 파일 (.dcm)
  - CT 스캔이면 수백~수천 장의 DICOM 파일 = 3D 볼륨 하나
  - 파일 안에 영상 데이터 + 환자 정보 + 촬영 정보가 함께 저장
  - 환자 이름, 생년월일, 진단명 등 민감 정보 포함
    → 딥러닝 활용 전 반드시 익명화(De-identification) 필요

PACS (서버):
  병원 내 모든 의료 영상을 저장, 관리, 전송하는 시스템
  방사선과 의사, 임상의가 영상을 조회하는 기반 인프라
```

```python
import pydicom
import numpy as np

# DICOM 파일 읽기
dcm = pydicom.dcmread('scan.dcm')

# 픽셀 데이터 추출
pixel_array = dcm.pixel_array   # (H, W) 또는 (슬라이스, H, W)

# 메타데이터 확인
print(dcm.PatientID)            # 환자 ID
print(dcm.Modality)             # CT, MRI, PET 등
print(dcm.PixelSpacing)         # 픽셀당 실제 크기 (mm)
print(dcm.SliceThickness)       # CT 슬라이스 두께 (mm)

# HU 값 변환 (CT의 경우)
hu = pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
```

### NIfTI (Neuroimaging Informatics Technology Initiative)

뇌 영상(MRI, fMRI) 연구에서 주로 사용하는 형식이다. DICOM과 달리 3D 볼륨 전체가 하나의 파일에 저장된다.

```
NIfTI 특징:
  - 3D 또는 4D(시계열) 볼륨 전체를 하나의 파일로 (.nii, .nii.gz)
  - 환자 개인 정보를 포함하지 않음 → 연구용으로 더 적합
  - 공간 정보(방향, 해상도)가 헤더에 저장

DICOM vs NIfTI:
  DICOM: 병원 임상 시스템 표준, 개인 정보 포함, 슬라이스별 파일
  NIfTI: 연구 목적 표준, 익명화됨, 볼륨 전체가 하나의 파일
```

```python
import nibabel as nib
import numpy as np

# NIfTI 파일 읽기
img = nib.load('brain.nii.gz')

# 3D 볼륨 데이터
volume = img.get_fdata()   # shape: (H, W, D) 또는 (H, W, D, T)
print(volume.shape)        # 예: (256, 256, 170) ← 170개 슬라이스

# 헤더 정보 (복셀 크기 등)
header = img.header
voxel_size = header.get_zooms()   # (1.0, 1.0, 1.0) mm 등
```

---

## 6. HU (Hounsfield Unit) — CT 값 척도

CT 영상의 픽셀 값은 **HU(Hounsfield Unit, 하운스필드 단위)** 로 표현된다. 서로 다른 조직의 밀도를 표준화된 수치로 나타낸 것이다. 딥러닝 모델을 위한 전처리에서 HU 값 범위를 이해하는 것이 중요하다.

|조직|HU 값 대략 범위|
|---|---|
|공기|-1000|
|지방|-950 ~ -400|
|지방|-100 ~ -50|
|물|0|
|근육/연부조직|30 ~ 100|
|뼈|300 ~ 1000 이상|

```python
import numpy as np

def windowing(hu_image, window_center, window_width):
    """
    CT 영상 윈도잉: 특정 조직에 맞게 HU 범위를 조정
    window_center: 보고자 하는 조직의 HU 중심값
    window_width:  HU 범위 너비
    """
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    windowed = np.clip(hu_image, lower, upper)
    # 0~1로 정규화
    return (windowed - lower) / window_width

# 폐 조직 보기: center=-600, width=1500
lung_window = windowing(hu_image, window_center=-600, window_width=1500)

# 뼈 보기: center=400, width=1800
bone_window = windowing(hu_image, window_center=400, window_width=1800)

# 복부 연부조직: center=40, width=400
soft_window = windowing(hu_image, window_center=40, window_width=400)
```

---

## 7. 의료 영상 전처리 파이프라인

의료 영상 딥러닝에서는 일반 이미지 데이터와 다른 특수한 전처리가 필요하다.

### Resampling / Respacing

CT, MRI 장비마다 촬영 해상도(복셀 크기)가 다르다. 어떤 장비는 1×1×1mm, 다른 장비는 0.5×0.5×2mm로 촬영한다. 딥러닝 모델은 일관된 입력 크기를 요구하므로, 모든 영상을 동일한 복셀 크기로 리샘플링해야 한다.

```python
import SimpleITK as sitk

def resample_volume(volume_path, new_spacing=[1.0, 1.0, 1.0]):
    """모든 CT를 1×1×1mm 복셀 크기로 리샘플링"""
    image = sitk.ReadImage(volume_path)
    original_spacing = image.GetSpacing()
    original_size    = image.GetSize()

    new_size = [
        int(round(orig_sz * orig_sp / new_sp))
        for orig_sz, orig_sp, new_sp
        in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)
```

### 주요 전처리 단계

**Cropping (크로핑)**: 관심 영역(ROI)만 잘라내어 불필요한 배경을 제거한다. 모델이 배경에 집중하지 않도록 한다.

**해상도 통일**: 환자마다 다르고, 기관마다 다른 해상도를 통일한다.

**HU 이상치 제거 및 빈공기 제거**: 모달리티(영상 장비 종류)에 따른 전처리. CT의 경우 관심 없는 HU 범위를 클리핑하거나, 환자 주변의 빈 공기 영역을 제거한다.

**원자별 정규화**: 환자마다 밝기가 다를 수 있어 각 환자(또는 슬라이스)별로 따로 정규화한다. Z-score 정규화(평균 0, 표준편차 1)가 의료 영상에서 많이 사용된다.

```python
def normalize_ct(image, lower=-1000, upper=1000):
    """CT HU 값 정규화"""
    image = np.clip(image, lower, upper)
    mean  = image.mean()
    std   = image.std()
    return (image - mean) / (std + 1e-8)   # Z-score

def normalize_mri(image):
    """MRI는 기관·장비마다 밝기 스케일이 다름 → 환자별 Z-score"""
    mean = image.mean()
    std  = image.std()
    return (image - mean) / (std + 1e-8)
```

**극심한 클래스 불균형 & 레이블 노이즈**: 의료 영상에서 종양 영역은 전체 이미지에서 1~5%에 불과한 경우가 많다. 배경 픽셀이 압도적으로 많아 모델이 항상 배경을 예측해도 높은 정확도가 나오는 문제가 있다.

```python
# 클래스 불균형 해결 — Dice Loss + Binary Cross Entropy 결합
def dice_loss(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])
    dice  = (2. * intersection + smooth) / (union + smooth)
    return 1 - tf.reduce_mean(dice)

def combined_loss(y_true, y_pred):
    bce  = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice
```

### 3D/4D 데이터 처리 전략

**Patch 기반 학습**: 512×512×300 크기의 3D CT 볼륨을 통째로 GPU에 올릴 수 없다. 작은 패치(예: 128×128×64)로 잘라서 학습하는 방식이 표준이다.

```python
def extract_patches_3d(volume, patch_size=(128, 128, 64), stride=(64, 64, 32)):
    """3D 볼륨에서 오버랩 패치 추출"""
    patches = []
    H, W, D = volume.shape
    pH, pW, pD = patch_size
    sH, sW, sD = stride

    for z in range(0, D - pD + 1, sD):
        for y in range(0, H - pH + 1, sH):
            for x in range(0, W - pW + 1, sW):
                patch = volume[y:y+pH, x:x+pW, z:z+pD]
                patches.append(patch)

    return np.array(patches)
```

---

## 8. 내시경 이미지 Segmentation 예시

폴립(용종) 탐지 등 내시경 이미지 분석에서 Segmentation이 활발하게 연구된다.

```
내시경 이미지 Segmentation 흐름:

원본 내시경 이미지
    ↓
전처리 (리사이즈, 정규화)
    ↓
U-Net / nnU-Net / UnetR
    ↓
Segmentation Mask 예측
    ↓
평가: IOU, Dice Coefficient
    ↓
Ground Truth와 비교 → 모델 성능 측정
```

내시경 데이터의 특수한 어려움:

- 카메라 각도, 조명에 따른 외관 변화
- 폴립의 다양한 크기와 모양
- 주름 등 배경과의 유사성
- 레이블링 비용이 매우 높음 (의사가 직접 마스크 그려야 함)

---

## 9. 핵심 요약

```
컴퓨터 비전 태스크 4단계:
  Classification          → 클래스만 (Single Object)
  Classification + Localize → 클래스 + Bounding Box
  Object Detection        → 여러 객체 + 각각 Bounding Box
  Segmentation            → 픽셀 단위 분류
    Semantic:   개체 구분 없이 클래스별 색칠
    Instance:   개체별 독립 마스크

평가 지표:
  IOU   = 겹치는 면적 / 합집합 면적  (0~1)
  Dice  = 2×겹치는 면적 / (예측면적 + 실제면적)  (의료 영상 특화)

U-Net 계열:
  U-Net     기본 구조: Contracting + Expansive + Skip Connection
  nnU-Net   자동 최적화 파이프라인, 의료 영상 벤치마크 표준
  UnetR     Transformer 인코더 + U-Net 디코더
  SwinUNet  Shifted Window Attention + U-Net

의료 영상 형식:
  DICOM   병원 표준, 슬라이스별 파일, 환자 정보 포함 → PACS 서버
  NIfTI   연구 표준, 3D 볼륨 하나의 파일, 익명화

HU (CT 값):
  공기=-1000, 물=0, 근육=30~100, 뼈=300~1000+

의료 영상 전처리 핵심:
  Resampling         → 복셀 크기 통일 (장비 간 차이 해소)
  Cropping           → ROI만 추출
  HU 이상치 제거     → 관심 없는 범위 클리핑
  원자별 정규화      → Z-score (환자별 밝기 차이 해소)
  클래스 불균형 대응 → Dice Loss + BCE 결합
  Patch 학습         → 3D 볼륨을 패치로 잘라서 학습
```


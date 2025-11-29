# -*- coding: utf-8 -*-
"""Colab-friendly human pose demo with fine-tuned classifier.

This script refactors the original demo01.py into a structured, end-to-end
pipeline that keeps the original MediaPipe Pose visualization/export features
while adding a small MLP classifier for pose-state recognition via transfer
learning.
"""

# ============================================================
# 0. 필수 라이브러리 설치 (Colab 전용)
#    - 필요 시, 아래 주석을 해제하고 실행하세요.
# ============================================================
# !pip install mediapipe opencv-python matplotlib numpy pandas scikit-learn tensorflow -q

# ============================================================
# 1. 라이브러리 임포트 & 전역 설정
# ============================================================
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Pose-state 라벨 목록 (확장 가능)
POSE_LABELS = ["standing", "sitting", "bending", "lying"]

# ============================================================
# 2. Colab용 업로드 유틸리티
# ============================================================
def upload_images():
    """Colab에서 여러 이미지를 업로드하고 파일 경로 리스트를 반환."""
    print("이미지 파일을 업로드하세요 (여러 개 선택 가능)")
    uploaded = files.upload()
    paths = list(uploaded.keys())
    if not paths:
        raise FileNotFoundError("업로드된 파일이 없습니다.")
    print(f"업로드된 파일: {paths}")
    return paths


# ============================================================
# 3. MediaPipe Pose 추론 (Transfer Learning: feature extractor)
# ============================================================
def run_mediapipe_pose(image_path):
    """이미지에서 MediaPipe Pose 추론 결과를 반환."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    image_bgr_copy = image_bgr.copy()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:
        results = pose.process(image_rgb)
    return image_bgr_copy, results


# ============================================================
# 4. 2D 오버레이 & 시각화
# ============================================================
def draw_2d_pose_overlay(image_bgr, results):
    if not results.pose_landmarks:
        print("[WARN] 포즈가 감지되지 않았습니다.")
        return image_bgr

    overlay = image_bgr.copy()
    mp_drawing.draw_landmarks(
        overlay,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )
    return overlay


def show_bgr_image_in_colab(image_bgr, title="Image"):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


# ============================================================
# 5. 3D 스켈레톤 시각화
# ============================================================
def extract_world_landmarks(results):
    """MediaPipe Pose 결과에서 3D world_landmarks 배열과 visibility 반환."""
    if not results.pose_world_landmarks:
        print("[WARN] 3D world_landmarks가 없습니다.")
        return None, None

    landmarks = results.pose_world_landmarks.landmark
    xs, ys, zs, vs = [], [], [], []
    for lm in landmarks:
        xs.append(lm.x)
        ys.append(lm.y)
        zs.append(lm.z)
        vs.append(lm.visibility)

    points_3d = np.array([xs, ys, zs]).T  # (33, 3)
    visibility = np.array(vs)  # (33,)
    return points_3d, visibility


def get_pose_connections():
    """MediaPipe Pose 33개 랜드마크의 스켈레톤 연결 정의."""
    connections = [
        (11, 12),  # 어깨
        (11, 13), (13, 15),  # 왼팔
        (12, 14), (14, 16),  # 오른팔
        (11, 23), (12, 24),  # 몸통
        (23, 24),  # 골반
        (23, 25), (25, 27), (27, 29), (29, 31),  # 왼 다리
        (24, 26), (26, 28), (28, 30), (30, 32),  # 오른 다리
    ]
    return connections


def plot_3d_skeleton(points_3d, visibility=None, vis_thresh=0.5):
    if points_3d is None:
        print("[ERROR] 3D 포인트가 없습니다.")
        return

    xs = points_3d[:, 0]
    ys = points_3d[:, 2]
    zs = -points_3d[:, 1]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, s=20, alpha=0.6)

    for i, j in get_pose_connections():
        if i >= len(points_3d) or j >= len(points_3d):
            continue
        if visibility is not None and (
            visibility[i] < vis_thresh or visibility[j] < vis_thresh
        ):
            continue
        x_line = [points_3d[i, 0], points_3d[j, 0]]
        y_line = [points_3d[i, 2], points_3d[j, 2]]
        z_line = [-points_3d[i, 1], -points_3d[j, 1]]
        ax.plot(x_line, y_line, z_line)

    ax.set_xlabel("X (left-right)")
    ax.set_ylabel("Y (depth)")
    ax.set_zlabel("Z (up-down)")

    max_range = np.array([
        xs.max() - xs.min(),
        ys.max() - ys.min(),
        zs.max() - zs.min(),
    ]).max()

    mid_x = (xs.max() + xs.min()) * 0.5
    mid_y = (ys.max() + ys.min()) * 0.5
    mid_z = (zs.max() + zs.min()) * 0.5

    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()
    plt.show()


# ============================================================
# 6. CSV 저장 (id,x,y,z,visibility)
# ============================================================
def save_pose_landmarks_to_csv(results, csv_path="data/keypoints/pose_sample.csv"):
    """pose_landmarks를 CSV로 저장."""
    if not results.pose_landmarks:
        print("[WARN] 저장할 pose_landmarks가 없습니다.")
        return None

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    rows = []
    for i, lm in enumerate(results.pose_landmarks.landmark):
        rows.append({
            "id": i,
            "x": lm.x,
            "y": lm.y,
            "z": lm.z,
            "visibility": lm.visibility,
        })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV 저장 완료: {csv_path}")
    return df


# ============================================================
# 7. 특징 추출: pose_landmarks -> 99D 벡터 (Transfer Learning)
# ============================================================
def pose_landmarks_to_vector(landmarks) -> np.ndarray:
    """33개 랜드마크의 (x,y,z) -> 길이 99 벡터(float32)."""
    if landmarks is None:
        raise ValueError("pose_landmarks가 없습니다.")
    coords = []
    for lm in landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    vec = np.array(coords, dtype=np.float32)
    if vec.shape[0] != 99:
        raise ValueError(f"벡터 길이가 99가 아닙니다: {vec.shape}")
    return vec


# ============================================================
# 8. 데이터셋 생성/수집 (pose_dataset.csv)
# ============================================================
def append_sample_to_dataset(vec, label, csv_path="pose_dataset.csv"):
    """단일 샘플을 pose_dataset.csv에 추가."""
    if vec.shape != (99,):
        raise ValueError("vec는 (99,) 형태여야 합니다.")

    columns = [f"v{i}" for i in range(99)] + ["label"]
    row = list(vec.astype(float)) + [label]
    write_header = not os.path.exists(csv_path)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    mode = "w" if write_header else "a"
    df = pd.DataFrame([row], columns=columns)
    df.to_csv(csv_path, mode=mode, header=write_header, index=False)
    print(f"데이터셋에 샘플 추가: label={label}, path={csv_path}")


def collect_samples_for_label(label):
    """주어진 라벨에 대해 여러 이미지를 업로드하여 pose_dataset.csv에 추가."""
    if label not in POSE_LABELS:
        print(f"[WARN] {label}가 POSE_LABELS에 없습니다. 그래도 진행합니다.")
    paths = upload_images()
    for path in paths:
        image_bgr, results = run_mediapipe_pose(path)
        if not results.pose_landmarks:
            print(f"[WARN] 포즈 미검출: {path}")
            continue
        vec = pose_landmarks_to_vector(results.pose_landmarks)
        append_sample_to_dataset(vec, label)

        overlay = draw_2d_pose_overlay(image_bgr, results)
        show_bgr_image_in_colab(overlay, title=f"{label} sample")


# ============================================================
# 9. 데이터셋 로드 및 전처리
# ============================================================
def load_pose_dataset(csv_path="pose_dataset.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"데이터셋 파일이 없습니다: {csv_path}")
    df = pd.read_csv(csv_path)
    feature_cols = [f"v{i}" for i in range(99)]
    X = df[feature_cols].values.astype(np.float32)
    y_text = df["label"].astype(str).values
    return X, y_text


# ============================================================
# 10. 분류기 정의 (Fine-tuning 대상)
# ============================================================
def build_mlp_classifier(input_dim, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_pose_classifier(csv_path="pose_dataset.csv"):
    """데이터셋을 로드하여 MLP 분류기를 학습/평가/저장."""
    X, y_text = load_pose_dataset(csv_path)
    le = LabelEncoder()
    y = le.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_mlp_classifier(input_dim=X.shape[1], num_classes=len(le.classes_))
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=16,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"테스트 정확도: {test_acc:.3f}, 손실: {test_loss:.3f}")

    model.save("pose_classifier_mlp.h5")
    np.save("label_classes.npy", le.classes_)
    print("모델과 라벨 인코더를 저장했습니다: pose_classifier_mlp.h5, label_classes.npy")
    return model, le, history


# ============================================================
# 11. 분류기 로드 및 추론
# ============================================================
def load_trained_classifier(model_path="pose_classifier_mlp.h5", classes_path="label_classes.npy"):
    if not os.path.exists(model_path) or not os.path.exists(classes_path):
        raise FileNotFoundError("학습된 모델 또는 클래스 파일이 없습니다.")
    model = keras.models.load_model(model_path)
    classes = np.load(classes_path)
    return model, classes


def predict_pose_label(image_path):
    """이미지에서 포즈 라벨을 예측하고 시각화."""
    model, classes = load_trained_classifier()
    image_bgr, results = run_mediapipe_pose(image_path)
    if not results.pose_landmarks:
        print("[WARN] 포즈를 감지하지 못했습니다.")
        return None

    vec = pose_landmarks_to_vector(results.pose_landmarks)
    preds = model.predict(vec.reshape(1, -1))
    top_idx = int(np.argmax(preds[0]))
    label = classes[top_idx]
    confidence = float(preds[0][top_idx])

    overlay = draw_2d_pose_overlay(image_bgr, results)
    show_bgr_image_in_colab(overlay, title=f"Pred: {label} ({confidence:.2f})")

    print(f"예측 라벨: {label}, 확신도: {confidence:.3f}")
    return label, confidence


# ============================================================
# 12. 데모 파이프라인 (기존 기능 유지)
# ============================================================
if __name__ == "__main__":
    # 1) 이미지 업로드
    paths = upload_images()
    image_path = paths[0]
    print("사용할 이미지:", image_path)

    # 2) Pose 추론
    image_bgr, results = run_mediapipe_pose(image_path)

    # 3) 2D 오버레이
    overlay_bgr = draw_2d_pose_overlay(image_bgr, results)
    show_bgr_image_in_colab(image_bgr, title="Input Image")
    show_bgr_image_in_colab(overlay_bgr, title="Pose Overlay (2D)")

    # 4) 3D 시각화
    points_3d, visibility = extract_world_landmarks(results)
    plot_3d_skeleton(points_3d, visibility=visibility, vis_thresh=0.5)

    # 5) CSV 저장 (id,x,y,z,visibility)
    save_pose_landmarks_to_csv(results, csv_path="data/keypoints/pose_sample.csv")

    # 6) 사용 가이드 출력
    print(
        """
사용 가이드 (Colab):
- 데이터 수집: collect_samples_for_label("standing") 등으로 라벨별 업로드
- 학습: train_pose_classifier()
- 추론: paths = upload_images(); predict_pose_label(paths[0])
- 원본 기능: 업로드된 이미지에 대해 2D/3D 시각화 및 CSV 저장
"""
    )

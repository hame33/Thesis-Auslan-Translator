import cv2
import mediapipe as mp
import numpy as np
import os

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def flatten_pose_landmarks(pose_landmarks):
    """
    Pose: 33 landmarks
    Each has x, y, z, visibility
    Output shape: 33 * 4 = 132
    """
    if pose_landmarks is None:
        return np.zeros(33 * 4, dtype=np.float32)

    features = []
    for lm in pose_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(features, dtype=np.float32)


def flatten_hand_landmarks(hand_landmarks):
    """
    Hand: 21 landmarks
    Each has x, y, z
    Output shape: 21 * 3 = 63
    """
    if hand_landmarks is None:
        return np.zeros(21 * 3, dtype=np.float32)

    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return np.array(features, dtype=np.float32)


def flatten_face_landmarks(face_landmarks, max_landmarks=468):
    """
    Face: 468 landmarks in MediaPipe Face Mesh
    Each has x, y, z
    Output shape: 468 * 3 = 1404
    """
    if face_landmarks is None:
        return np.zeros(max_landmarks * 3, dtype=np.float32)

    features = []
    for lm in face_landmarks.landmark[:max_landmarks]:
        features.extend([lm.x, lm.y, lm.z])

    # pad just in case
    expected_len = max_landmarks * 3
    if len(features) < expected_len:
        features.extend([0.0] * (expected_len - len(features)))

    return np.array(features, dtype=np.float32)


def extract_features_from_video(video_path, save_path=None, draw=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    all_features = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # BGR -> RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run landmark detection
            results = holistic.process(rgb)

            # Flatten each modality
            pose_feat = flatten_pose_landmarks(results.pose_landmarks)
            left_hand_feat = flatten_hand_landmarks(results.left_hand_landmarks)
            right_hand_feat = flatten_hand_landmarks(results.right_hand_landmarks)
            face_feat = flatten_face_landmarks(results.face_landmarks)

            # Concatenate into one frame feature vector
            frame_features = np.concatenate(
                [pose_feat, left_hand_feat, right_hand_feat, face_feat],
                axis=0
            )

            all_features.append(frame_features)

            if draw:
                annotated = frame.copy()

                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS
                )
                mp_drawing.draw_landmarks(
                    annotated,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )
                mp_drawing.draw_landmarks(
                    annotated,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )
                mp_drawing.draw_landmarks(
                    annotated,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS
                )

                cv2.imshow("Landmarks", annotated)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

    cap.release()
    cv2.destroyAllWindows()

    if len(all_features) == 0:
        raise ValueError("No frames processed from video.")

    features_array = np.stack(all_features)  # [num_frames, feature_dim]

    if save_path is not None:
        np.save(save_path, features_array)
        print(f"Saved features to: {save_path}")

    print(f"Feature shape: {features_array.shape}")
    return features_array


if __name__ == "__main__":
    video_path = "input.mp4"
    save_path = "input_features.npy"

    features = extract_features_from_video(
        video_path=video_path,
        save_path=save_path,
        draw=False
    )
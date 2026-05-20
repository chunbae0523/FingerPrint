import time
import cv2
import math  # 거리 계산을 위해 추가
import mediapipe as mp
import os
import sys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 손가락 마디들을 연결할 순서 정의
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # 엄지
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # 검지
    (9, 10),
    (10, 11),
    (11, 12),  # 중지
    (13, 14),
    (14, 15),
    (15, 16),  # 약지
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # 새끼
    (5, 9),
    (9, 13),
    (13, 17),  # 손바닥 안쪽
]

latest_result = None


# 비동기 콜백 함수
def print_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


# --- 색상 및 폰트 변수 정의 ---
COLOR_POINT = (0, 255, 255)  # 노란색 (점)
COLOR_LINE = (255, 0, 0)  # 파란색 (선)
COLOR_TEXT = (0, 255, 0)  # 초록색 (숫자 인덱스)
COLOR_BG = (0, 0, 0)  # 검은색 (텍스트 배경)
COLOR_SIGN = (0, 255, 0)  # 초록색 (번역된 텍스트 강조)


# --- 두 점 사이의 거리를 구하는 함수 ---
def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# --- 수화/지문자 인식 함수 ---
def recognize_gesture(pixel_landmarks):
    if len(pixel_landmarks) < 21:
        return "Unknown"

    # 각 손가락이 펼쳐졌는지 여부 (True = 펼쳐짐, False = 접힘)
    fingers = [False] * 5

    # 1. 엄지 손가락 판별 (X축 기준으로 마디가 확장되었는지 확인)
    if abs(pixel_landmarks[4][0] - pixel_landmarks[0][0]) > abs(
        pixel_landmarks[3][0] - pixel_landmarks[0][0]
    ):
        fingers[0] = True

    # 2. 나머지 네 손가락 판별 (Y축 기준, Tip이 PIP 마디보다 위에 있으면 펼쳐진 것)
    if pixel_landmarks[8][1] < pixel_landmarks[6][1]:
        fingers[1] = True  # 검지
    if pixel_landmarks[12][1] < pixel_landmarks[10][1]:
        fingers[2] = True  # 중지
    if pixel_landmarks[16][1] < pixel_landmarks[14][1]:
        fingers[3] = True  # 약지
    if pixel_landmarks[20][1] < pixel_landmarks[18][1]:
        fingers[4] = True  # 새끼

    # 3. 특수 제스처 처리를 위한 거리 및 위치 계산
    # 엄지 끝(4번)과 검지 끝(8번)의 거리
    thumb_index_dist = get_distance(pixel_landmarks[4], pixel_landmarks[8])

    # 손가락 한 마디 정도의 기준 거리 계산 (손의 크기에 맞춰 유동적으로 변하도록 5번-17번 거리 활용)
    palm_size = get_distance(pixel_landmarks[5], pixel_landmarks[17])

    # --- 4. 제스처 조건 판단 ---

    # [추가] OK 모양: 엄지와 검지 끝이 만나고(거리가 가깝고) 나머지 세 손가락은 펴진 상태
    if (
        thumb_index_dist < (palm_size * 0.3)
        and fingers[2]
        and fingers[3]
        and fingers[4]
    ):
        return "OK"

    # [추가] 따봉(Thumbs Up): 엄지만 펴지고 대다수 접힘 + 엄지 끝이 엄지 시작점(2번 마디)보다 위에 있음
    if (
        fingers[0]
        and not fingers[1]
        and not fingers[2]
        and not fingers[3]
        and not fingers[4]
    ):
        if pixel_landmarks[4][1] < pixel_landmarks[2][1]:
            return "Thumbs Up (Good!)"

    # 기존 매핑 리스트
    if fingers == [False, True, False, False, False]:
        return "1 (Num 1)"
    elif fingers == [False, True, True, False, False]:
        return "2 (Num 2) / V"
    elif fingers == [False, True, True, True, False]:
        return "3 (Num 3)"
    elif fingers == [False, True, True, True, True]:
        return "4 (Num 4)"
    elif fingers == [True, True, True, True, True]:
        return "5 (Num 5) / Hello"
    elif fingers == [True, True, False, False, False]:
        return "L (Alphabet L)"
    elif fingers == [True, False, False, False, True]:
        return "Spider-Man / I Love You"
    elif fingers == [False, False, False, False, True]:
        return "Pinky (Promise)"
    elif fingers == [False, False, False, False, False]:
        return "Fist (Rock)"

    return "Analyzing..."


# --- [EXE 빌드를 위한 경로 파일 탐색 함수] ---
def get_resource_path(relative_path):
    """PyInstaller 임시 폴더 또는 main.py 파일이 있는 디렉토리에서 절대 경로를 반환합니다."""
    # 1. PyInstaller로 빌드된 파일인 경우 임시 폴더(_MEIPASS) 경로 반환
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)

    # 2. 일반 스크립트 실행인 경우, 터미널 위치와 무관하게 현재 파일(main.py)의 폴더 경로 기준
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_file_dir, relative_path)


model_path = get_resource_path("hand_landmarker.task")
base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=print_result,
)

cap = cv2.VideoCapture(0)

with vision.HandLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        timestamp = int(time.time() * 1000)
        detector.detect_async(mp_image, timestamp)

        # 2. 결과 시각화 및 수화 번역
        if latest_result and latest_result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(latest_result.hand_landmarks):
                pixel_landmarks = []
                for idx, landmark in enumerate(hand_landmarks):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    pixel_landmarks.append((cx, cy))

                    cv2.circle(frame, (cx, cy), 4, COLOR_POINT, cv2.FILLED)
                    cv2.putText(
                        frame,
                        str(idx),
                        (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        COLOR_TEXT,
                        1,
                        cv2.LINE_AA,
                    )

                for start_idx, end_idx in HAND_CONNECTIONS:
                    if start_idx < len(pixel_landmarks) and end_idx < len(
                        pixel_landmarks
                    ):
                        cv2.line(
                            frame,
                            pixel_landmarks[start_idx],
                            pixel_landmarks[end_idx],
                            COLOR_LINE,
                            2,
                        )

                # 제스처 결과 분석
                gesture_text = recognize_gesture(pixel_landmarks)

                # 손목(0번) 기준 아래쪽에 텍스트 출력
                text_pos = (pixel_landmarks[0][0] - 20, pixel_landmarks[0][1] + 40)
                cv2.rectangle(
                    frame,
                    (text_pos[0] - 5, text_pos[1] - 25),
                    (text_pos[0] + 280, text_pos[1] + 10),
                    COLOR_BG,
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    f"Hand {hand_idx + 1}: {gesture_text}",
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    COLOR_SIGN,
                    2,
                    cv2.LINE_AA,
                )

        cv2.putText(
            frame,
            "Sign Language Translator",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Custom MediaPipe Hands", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

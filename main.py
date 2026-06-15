import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from modules.utils import Vector_Normalization
from PIL import ImageFont, ImageDraw, Image
from modules.unicode import join_jamos  # 자모 결합용

# --- 설정 (기존 설정 유지) ---
COLOR_LANDMARK = (170, 170, 255)
COLOR_CONNECTION = (190, 190, 255)
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

fontpath = "Korean.TTF"
font = ImageFont.truetype(fontpath, 50)
actions = [
    "ㄱ",
    "ㄴ",
    "ㄷ",
    "ㄹ",
    "ㅁ",
    "ㅂ",
    "ㅅ",
    "ㅇ",
    "ㅈ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
    "ㅏ",
    "ㅑ",
    "ㅓ",
    "ㅕ",
    "ㅗ",
    "ㅛ",
    "ㅜ",
    "ㅠ",
    "ㅡ",
    "ㅣ",
    "ㅐ",
    "ㅒ",
    "ㅔ",
    "ㅖ",
    "ㅢ",
    "ㅚ",
    "ㅟ",
]


# --- 윤곽선 함수 추가 ---
def draw_text_with_outline(
    draw, position, text, font, fill_color, outline_color, outline_width=2
):
    x, y = position
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    draw.text((x, y), text, font=font, fill=fill_color)


# --- 변수 초기화 ---
final_sentence = []
last_action = None
frame_count = 0
seq = []
last_word = None

# 모델 로드 (기존 유지)
base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

interpreter = tf.lite.Interpreter(
    model_path="models/multi_hand_gesture_classifier.tflite"
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    results = detector.detect(mp_image)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    if results.hand_landmarks:
        landmarks = results.hand_landmarks[0]

        # 시각화 로직 (기존 유지)
        for start, end in HAND_CONNECTIONS:
            p1 = (
                int(landmarks[start].x * frame.shape[1]),
                int(landmarks[start].y * frame.shape[0]),
            )
            p2 = (
                int(landmarks[end].x * frame.shape[1]),
                int(landmarks[end].y * frame.shape[0]),
            )
            draw.line((p1, p2), fill=COLOR_CONNECTION, width=3)
        for lm in landmarks:
            x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            draw.ellipse(
                (x - 3, y - 3, x + 3, y + 3), fill=COLOR_LANDMARK, outline="white"
            )

        # 모델 예측 및 문장 조합 로직
        joint = np.array([[lm.x, lm.y] for lm in landmarks])
        vector, angle_label = Vector_Normalization(joint)
        seq.append(np.concatenate([vector.flatten(), angle_label.flatten()]))
        if len(seq) > 10:
            seq.pop(0)

        if len(seq) == 10:
            input_data = np.expand_dims(np.array(seq, dtype=np.float32), axis=0)
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()
            y_pred = interpreter.get_tensor(output_details[0]["index"])
            i_pred = int(np.argmax(y_pred[0]))

            if y_pred[0][i_pred] > 0.7:
                this_action = actions[i_pred]
                if last_action == this_action and last_word != this_action:
                    frame_count += 1
                    bar_width = (frame_count / 20) * 50
                    draw.rectangle(
                        (20, 70, 20 + bar_width, 80),
                        fill=(237, 255, 133),
                    )
                else:
                    frame_count = 0
                last_action = this_action
                draw_text_with_outline(
                    draw,
                    (20, 20),
                    this_action,
                    font,
                    fill_color=(237, 255, 133),
                    outline_color=(50, 50, 50),  # 짙은 회색 테두리
                    outline_width=2,  # 테두리 두께
                )

                if frame_count == 20:
                    final_sentence.append(this_action)
                    last_word = this_action
                    frame_count = 0

    # 하단 텍스트 출력
    sentence_text = join_jamos("".join(final_sentence))
    draw_text_with_outline(
        draw,
        (40, frame.shape[0] - 75),
        sentence_text,
        font,
        (255, 255, 255),
        (0, 0, 0),
        2,
    )

    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow("Sign Language Translator", frame)

    key = cv2.waitKey(1) & 0xFF
    # ESC 키로 종료
    if key == 27:
        break
    # 백스페이스 키(8)를 눌렀을 때 마지막 글자 삭제
    elif key == 8:
        if len(final_sentence) > 0:
            final_sentence.pop()
            last_word = None

cap.release()
cv2.destroyAllWindows()

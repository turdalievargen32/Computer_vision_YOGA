import cv2
import mediapipe as mp
import numpy as np
import joblib

# Загрузка модели
model = joblib.load('pose_classifier.pkl')

# MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Камера
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Обработка изображения
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # Рисуем скелет
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark
        row = []
        for lm in landmarks:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])

        if len(row) == 132:  # 33 точки * 4 координаты
            X_input = np.array(row).reshape(1, -1)
            prediction = model.predict(X_input)[0]
            proba = model.predict_proba(X_input).max() * 100

            # Отображаем предсказание
            COLOR_GOOD = (0, 255, 0)    # Зелёный
            COLOR_BAD = (0, 0, 255)     # Красный
            FONT = cv2.FONT_HERSHEY_SIMPLEX
            CONFIDENCE_THRESHOLD = 40  # Можно менять по желанию

            text_color = COLOR_GOOD if proba >= CONFIDENCE_THRESHOLD else COLOR_BAD

            if proba < CONFIDENCE_THRESHOLD:
                    prediction = "Unknown"

            pose_text = f'Pose: {prediction}'
            conf_text = f'Confidence: {proba:.2f}%'

            cv2.putText(image, pose_text, (10, 40), FONT, 1.0, text_color, 2, cv2.LINE_AA)
            cv2.putText(image, conf_text, (10, 80), FONT, 1.0, text_color, 2, cv2.LINE_AA)


    cv2.imshow('Pose Detection', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




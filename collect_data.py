import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import argparse
import time

# Аргументы командной строки
parser = argparse.ArgumentParser()
parser.add_argument('--pose', type=str, required=True, help='Pose label (e.g., tree, neutral)')
parser.add_argument('--count', type=int, default=50, help='Number of samples to collect')
parser.add_argument('--delay', type=float, default=2.0, help='Delay between captures (in seconds)')
args = parser.parse_args()

POSE_LABEL = args.pose
SAMPLES = args.count
DELAY = args.delay
CSV_FILE = 'data.csv'

# Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Подготовка DataFrame
feature_names = [f'{i}_{axis}' for i in range(33) for axis in ['x', 'y', 'z', 'v']]
columns = ['label'] + feature_names

if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
else:
    df = pd.DataFrame(columns=columns)

# Камера
cap = cv2.VideoCapture(0)
collected = 0

print(f'📷 Начинаем сбор данных для позы "{POSE_LABEL}". Подготовьтесь...')
time.sleep(3)

last_time = time.time()

while cap.isOpened() and collected < SAMPLES:
    ret, frame = cap.read()
    if not ret:
        continue

    # Обработка кадра
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Рисуем скелет
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Захват с паузой
        if time.time() - last_time > DELAY:
            landmarks = results.pose_landmarks.landmark
            row = [POSE_LABEL]
            for lm in landmarks:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])

            if len(row) == len(columns):
                df.loc[len(df)] = row
                collected += 1
                print(f'✅ Сохранено: {collected}/{SAMPLES}')
            else:
                print('⚠️ Неправильная длина строки данных.')

            last_time = time.time()

    cv2.putText(image, f'Pose: {POSE_LABEL} | Collected: {collected}/{SAMPLES}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Collecting Data', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Сохраняем
df.to_csv(CSV_FILE, index=False)
cap.release()
cv2.destroyAllWindows()
print('📁 Данные сохранены в', CSV_FILE)

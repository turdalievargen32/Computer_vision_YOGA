import pandas as pd
df = pd.read_csv('data.csv')
print(df['label'].value_counts())


import cv2, mediapipe as mp, numpy as np, joblib, imageio, time, os

POSE_DIR = 'poses_media'
POSE_SEQUENCE = ['neutral', 'tree', 'warrior']  # Изменено 'neural' на 'neutral'
POSE_TIME = 30
CONF_THRESHOLD = 40

print("Loading media files...")
MEDIA = {}
def load_media(name):
    for ext in ('.gif', '.png', '.jpg', '.jpeg'):
        path = os.path.join(POSE_DIR, name + ext)
        if os.path.exists(path):
            break
    else:
        print(f'Warning: File for pose "{name}" not found, creating placeholder')
        placeholder = np.zeros((320, 220, 3), dtype=np.uint8)
        cv2.putText(placeholder, name.upper(), (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return [placeholder]

    if path.endswith('.gif'):
        try:
            frames = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), (220, 320)) for f in imageio.mimread(path)]
            return frames
        except:
            img = cv2.imread(path)
            return [cv2.resize(img, (220, 320))] if img is not None else [placeholder]
    else:
        img = cv2.imread(path)
        return [cv2.resize(img, (220, 320))] if img is not None else [placeholder]

for pose in POSE_SEQUENCE:
    MEDIA[pose] = load_media(pose)
    print(f"Pose '{pose}' loaded")

print("Loading AI model...")
try:
    model = joblib.load('pose_classifier.pkl')
    print("Model loaded successfully")
except:
    print("Model loading error! Check pose_classifier.pkl")
    exit()

mp_pose, mp_draw = mp.solutions.pose, mp.solutions.drawing_utils
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not connect to camera!")
    exit()

print("Camera connected")

def draw_text_with_bg(img, text, pos, font_scale=1, color=(255,255,255), bg_color=(0,0,0), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (pos[0] - 5, pos[1] - text_height - 5), (pos[0] + text_width + 5, pos[1] + baseline + 5), bg_color, -1)
    cv2.putText(img, text, pos, font, font_scale, color, thickness)

def center_text(img, text, y, font_scale, color, thickness=2):
    """Центрирует текст по горизонтали"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (img.shape[1] - text_size[0]) // 2
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def run_pose_game():
    print("\nYOGA POSE GAME STARTING!")
    print("Each pose lasts 30 seconds. Perform the pose correctly with at least 40% confidence.")
    print("You will earn 1 point per successful pose. ESC or Q to quit.\n")

    score = 0
    total = len(POSE_SEQUENCE)
    results = {}  # Сохраняем результаты для каждой позы

    # Показываем вступительный экран
    show_intro_screen()

    for pose_name in POSE_SEQUENCE:
        print(f"Starting pose: {pose_name.upper()} ({POSE_TIME} seconds)")
        show_transition_screen(pose_name, POSE_TIME)
        continue_game, pose_completed = execute_pose_round(pose_name)

        if not continue_game:
            print("Game interrupted by user.")
            return

        results[pose_name] = pose_completed
        if pose_completed:
            print(f"Pose {pose_name} completed! +1 point")
            score += 1
            show_pose_result(True, pose_name, earned_point=True)
        else:
            print(f"Pose {pose_name} failed. 0 points")
            show_pose_result(False, pose_name, earned_point=False)

    show_final_results(score, total, results)

def show_intro_screen():
    """Показывает вступительный экран"""
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Создаем темный overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.2, overlay, 0.8, 0)
        
        # Добавляем текст
        center_text(frame, "YOGA POSE GAME", h//2 - 80, 1.5, (0, 255, 255), 3)
        center_text(frame, "Perform 3 poses correctly", h//2 - 40, 1.0, (255, 255, 255))
        center_text(frame, "Neutral -> Tree -> Warrior", h//2, 0.9, (255, 255, 0))
        center_text(frame, "Minimum 40% confidence required", h//2 + 40, 0.8, (255, 255, 255))
        center_text(frame, "Game starts in 3 seconds...", h//2 + 80, 0.7, (0, 255, 0))
        
        cv2.imshow('Yoga Poses Game', frame)
        cv2.waitKey(3000)

def show_transition_screen(next_pose_name, time_limit):
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)

        # Используем центрированный текст
        center_text(frame, "NEXT POSE", h//2 - 60, 1.2, (255, 255, 255))
        
        # Цвет для каждой позы
        pose_colors = {
            'neutral': (128, 128, 255),  # Светло-фиолетовый
            'tree': (0, 255, 0),         # Зеленый
            'warrior': (0, 165, 255)     # Оранжевый
        }
        pose_color = pose_colors.get(next_pose_name, (0, 255, 255))
        
        center_text(frame, next_pose_name.upper(), h//2, 2.0, pose_color, 3)
        center_text(frame, f"Time: {time_limit} seconds", h//2 + 40, 1.0, (255, 255, 0))
        center_text(frame, "Get ready... Game starts in 3 seconds", h//2 + 80, 0.7, (255, 255, 255))

        cv2.imshow('Yoga Poses Game', frame)
        cv2.waitKey(3000)

def execute_pose_round(target_pose):
    frames = MEDIA[target_pose]
    f_idx = 0
    matched = False
    start_time = time.time()

    while time.time() - start_time < POSE_TIME:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.pose_landmarks:
            mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            row = []
            for lm in res.pose_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])

            if len(row) == 132:
                X = np.array(row).reshape(1, -1)
                proba = model.predict_proba(X)[0]
                pred = model.classes_[np.argmax(proba)]
                conf = np.max(proba) * 100

                draw_text_with_bg(frame, f"{pred} {conf:.1f}%", (10, 50), 0.9, (0,255,0) if conf >= CONF_THRESHOLD else (0,165,255))

                if pred == target_pose and conf >= CONF_THRESHOLD:
                    matched = True
                    break

        # Показываем референсное изображение
        ref = frames[f_idx]
        f_idx = (f_idx + 1) % len(frames)
        frame[10:10+ref.shape[0], w - ref.shape[1] - 10:w - 10] = ref

        # Таймер
        remaining = max(0, POSE_TIME - (time.time() - start_time))
        timer_color = (0, 255, 0) if remaining > 10 else (0, 165, 255) if remaining > 5 else (0, 0, 255)
        draw_text_with_bg(frame, f"Time: {remaining:.1f}s", (w - 200, h - 30), 0.8, timer_color)

        # Информация о цели
        draw_text_with_bg(frame, f"Target: {target_pose.upper()}", (10, h - 40), 0.8, (255, 255, 255))
        draw_text_with_bg(frame, f"Required confidence: {CONF_THRESHOLD}%", (10, h - 70), 0.7, (200, 200, 200))

        cv2.imshow("Yoga Pose Game", frame)
        key = cv2.waitKey(30) & 0xFF
        if key in [27, ord('q')]:
            return False, False

    return True, matched

def show_pose_result(success, pose_name, earned_point=False):
    ret, frame = cap.read()
    if not ret:
        return
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
    
    color = (0, 255, 0) if success else (0, 0, 255)
    status = "SUCCESS" if success else "FAILED"
    points = "+1 POINT" if earned_point else "0 POINTS"
    
    center_text(frame, f"{pose_name.upper()}", h//2 - 40, 1.5, (255, 255, 255), 3)
    center_text(frame, status, h//2, 1.2, color, 3)
    center_text(frame, points, h//2 + 40, 1.0, color)
    
    cv2.imshow("Yoga Pose Game", frame)
    cv2.waitKey(2000)

def show_final_results(score, total, results):
    """Показывает финальные результаты с простым фидбеком"""
    ret, frame = cap.read()
    if not ret:
        return
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Создаем темный фон
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.2, overlay, 0.8, 0)
    
    # Заголовок
    center_text(frame, "FINAL RESULTS", h//2 - 80, 1.8, (0, 255, 255), 3)
    
    # Основной счет
    score_color = (0, 255, 0) if score == total else (255, 255, 0) if score > 0 else (0, 0, 255)
    center_text(frame, f"SCORE: {score} / {total}", h//2 - 20, 1.5, score_color, 3)
    
    # Простой фидбек
    if score == 0:
        feedback = "Keep practicing!"
        feedback_color = (0, 165, 255)
    elif score == 1:
        feedback = "Keep going!"
        feedback_color = (255, 255, 0)
    elif score == 2:
        feedback = "Almost perfect!"
        feedback_color = (0, 255, 0)
    else:  # score == 3
        feedback = "Perfect score!"
        feedback_color = (0, 255, 0)
    
    center_text(frame, feedback, h//2 + 30, 1.2, feedback_color, 2)
    
    # Инструкция для выхода
    center_text(frame, "Press any key to exit...", h//2 + 80, 0.6, (255, 255, 255))
    
    cv2.imshow("Yoga Pose Game", frame)
    cv2.waitKey(0)  # Ждем нажатия любой клавиши

def main():
    """Главная функция"""
    try:
        run_pose_game()
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released. Goodbye!")

# Запуск игры
if __name__ == "__main__":
    main()
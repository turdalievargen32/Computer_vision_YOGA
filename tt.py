import pandas as pd
df = pd.read_csv('data.csv')
print(df['label'].value_counts())


import cv2, mediapipe as mp, numpy as np, joblib, imageio, time, os

POSE_DIR = 'poses_media'
POSE_SEQUENCE = ['neural', 'tree', 'warrior']
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

def run_pose_game():
    print("\nYOGA POSE GAME STARTING!")
    print("Each pose lasts 30 seconds. Perform the pose correctly with at least 40% confidence.")
    print("You will earn 1 point per successful pose. ESC or Q to quit.\n")

    score = 0
    total = len(POSE_SEQUENCE)

    for pose_name in POSE_SEQUENCE:
        print(f"Starting pose: {pose_name.upper()} ({POSE_TIME} seconds)")
        show_transition_screen(pose_name, POSE_TIME)
        continue_game, pose_completed = execute_pose_round(pose_name)

        if not continue_game:
            print("Game interrupted by user.")
            return

        if pose_completed:
            print(f"Pose {pose_name} completed! +1 point")
            score += 1
            show_pose_result(True, pose_name, earned_point=True)
        else:
            print(f"Pose {pose_name} failed. 0 points")
            show_pose_result(False, pose_name, earned_point=False)

    show_final_results(score, total)

def show_transition_screen(next_pose_name, time_limit):
    ret, transition_frame = cap.read()
    if ret:
        transition_frame = cv2.flip(transition_frame, 1)
        h, w = transition_frame.shape[:2]
        overlay = transition_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        transition_frame = cv2.addWeighted(transition_frame, 0.3, overlay, 0.7, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        title = "NEXT POSE"
        (tw, th), _ = cv2.getTextSize(title, font, 1.2, 2)
        cv2.putText(transition_frame, title, ((w-tw)//2, h//2-60), font, 1.2, (255, 255, 255), 2)

        pose_name = next_pose_name.upper()
        (tw, th), _ = cv2.getTextSize(pose_name, font, 2.0, 3)
        cv2.putText(transition_frame, pose_name, ((w-tw)//2, h//2), font, 2.0, (0, 255, 255), 3)

        time_text = f"Time: {time_limit} seconds"
        (tw, th), _ = cv2.getTextSize(time_text, font, 1.0, 2)
        cv2.putText(transition_frame, time_text, ((w-tw)//2, h//2+40), font, 1.0, (255, 255, 0), 2)

        instruction = "Get ready... Game starts in 3 seconds"
        (tw, th), _ = cv2.getTextSize(instruction, font, 0.7, 2)
        cv2.putText(transition_frame, instruction, ((w-tw)//2, h//2+80), font, 0.7, (255, 255, 255), 2)

        cv2.imshow('Yoga Poses Game', transition_frame)
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

        ref = frames[f_idx]
        f_idx = (f_idx + 1) % len(frames)
        frame[10:10+ref.shape[0], w - ref.shape[1] - 10:w - 10] = ref

        remaining = max(0, POSE_TIME - (time.time() - start_time))
        timer_color = (0, 255, 0) if remaining > 10 else (0, 165, 255) if remaining > 5 else (0, 0, 255)
        draw_text_with_bg(frame, f"Time: {remaining:.1f}s", (w - 200, h - 30), 0.8, timer_color)

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
    color = (0, 255, 0) if success else (0, 165, 255)
    message = f"{pose_name.upper()} {'+1 POINT' if earned_point else 'FAILED'}"
    draw_text_with_bg(frame, message, (w//4, h//2), 1.2, color)
    cv2.imshow("Yoga Pose Game", frame)
    cv2.waitKey(2000)

def show_final_results(score, total):
    ret, frame = cap.read()
    if not ret:
        return
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.2, overlay, 0.8, 0)
    draw_text_with_bg(frame, "FINAL RESULTS", (w//3, h//2 - 60), 1.5, (255, 255, 255))
    draw_text_with_bg(frame, f"SCORE: {score} / {total}", (w//3, h//2), 1.3, (0, 255, 255))
    cv2.imshow("Yoga Pose Game", frame)
    cv2.waitKey(5000)

try:
    run_pose_game()
except KeyboardInterrupt:
    print("\nGame interrupted")
except Exception as e:
    print(f"\nError: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released. Goodbye!")

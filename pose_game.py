import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import random
import os

class PoseGame:
    def __init__(self, model_path='pose_classifier.pkl'):
        # Проверяем существование модели
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        # Загружаем модель
        self.model = joblib.load(model_path)
        self.labels = self.model.classes_.tolist()
        print(f"Загружена модель с классами: {self.labels}")
        
        # Настройки игры
        self.target_duration = 50  # Секунды на выполнение позы
        self.confidence_threshold = 50  # Минимальный % для засчитывания (понижен с 80 до 50)
        self.rounds = 3
        
        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Инициализация камеры
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Не удалось открыть камеру")
        
        # Устанавливаем разрешение камеры
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def get_landmarks(self, results):
        """Извлекаем координаты ключевых точек"""
        if not results.pose_landmarks:
            return None
        
        row = []
        for lm in results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
        return row
    
    def predict_pose(self, landmarks):
        """Предсказываем позу по ключевым точкам"""
        if landmarks is None or len(landmarks) != 132:
            return None, 0
        
        try:
            X_input = np.array(landmarks).reshape(1, -1)
            prediction = self.model.predict(X_input)[0]
            confidence = self.model.predict_proba(X_input).max() * 100
            return prediction, confidence
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return None, 0
    
    def draw_text(self, image, text, position, color=(255, 255, 255), scale=1.0, thickness=2):
        """Рисуем текст с обводкой для лучшей читаемости"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Черная обводка
        cv2.putText(image, text, position, font, scale, (0, 0, 0), thickness + 2)
        # Основной текст
        cv2.putText(image, text, position, font, scale, color, thickness)
    
    def play_round(self, round_num, target_pose):
        """Играем один раунд"""
        print(f"\nРаунд {round_num}: Покажите позу '{target_pose}'")
        
        start_time = time.time()
        pose_detected = False
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Ошибка чтения кадра с камеры")
                break
            
            # Отражаем изображение для удобства
            frame = cv2.flip(frame, 1)
            
            # Конвертируем в RGB для MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            
            # Возвращаем в BGR для OpenCV
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Вычисляем оставшееся время
            elapsed_time = time.time() - start_time
            remaining_time = max(0, self.target_duration - elapsed_time)
            
            # Рисуем интерфейс
            self.draw_text(image, f"Round {round_num}/{self.rounds}", (10, 30), (255, 255, 255), 0.8)
            self.draw_text(image, f"Target: {target_pose}", (10, 60), (0, 255, 255), 1.0, 2)
            self.draw_text(image, f"Time: {remaining_time:.1f}s", (10, 90), (255, 255, 255), 0.8)
            
            # Обрабатываем позу
            if results.pose_landmarks:
                # Рисуем скелет
                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )
                
                # Получаем предсказание
                landmarks = self.get_landmarks(results)
                prediction, confidence = self.predict_pose(landmarks)
                
                if prediction:
                    # Проверяем правильность позы
                    if prediction == target_pose and confidence > self.confidence_threshold:
                        self.draw_text(image, "✅ CORRECT!", (10, 130), (0, 255, 0), 1.2, 3)
                        pose_detected = True
                    elif prediction == target_pose:
                        # Правильная поза, но низкая уверенность
                        self.draw_text(image, f"⚠️ {prediction} (Low confidence)", (10, 130), (255, 165, 0), 1.0)
                        self.draw_text(image, f"Confidence: {confidence:.1f}% (need {self.confidence_threshold}%)", (10, 160), (255, 165, 0), 0.7)
                    else:
                        self.draw_text(image, f"❌ Detected: {prediction}", (10, 130), (0, 0, 255), 1.0)
                        self.draw_text(image, f"Confidence: {confidence:.1f}%", (10, 160), (255, 255, 255), 0.7)
            else:
                self.draw_text(image, "⚠️ No pose detected", (10, 130), (255, 165, 0), 1.0)
            
            # Показываем изображение
            cv2.imshow('Pose Game - Press Q to quit', image)
            
            # Проверяем условия выхода
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False, True  # Не засчитан, выход из игры
            
            if pose_detected:
                # Показываем успех еще секунду
                success_start = time.time()
                while time.time() - success_start < 1.0:
                    ret, frame = self.cap.read()
                    if ret:
                        frame = cv2.flip(frame, 1)
                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = self.pose.process(image_rgb)
                        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                        
                        if results.pose_landmarks:
                            self.mp_drawing.draw_landmarks(
                                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                            )
                        
                        self.draw_text(image, f"Round {round_num}/{self.rounds}", (10, 30), (255, 255, 255), 0.8)
                        self.draw_text(image, f"Target: {target_pose}", (10, 60), (0, 255, 255), 1.0, 2)
                        self.draw_text(image, "🎉 SUCCESS! 🎉", (10, 130), (0, 255, 0), 1.5, 3)
                        
                        cv2.imshow('Pose Game - Press Q to quit', image)
                        cv2.waitKey(1)
                
                return True, False  # Засчитан, продолжаем игру
            
            if remaining_time <= 0:
                return False, False  # Время вышло, не засчитан
    
    def play(self):
        """Основная игра"""
        print("=== POSE GAME STARTED ===")
        print(f"Доступные позы: {', '.join(self.labels)}")
        print(f"Раундов: {self.rounds}")
        print(f"Время на позу: {self.target_duration} секунд")
        print(f"Минимальная уверенность: {self.confidence_threshold}%")
        print("Нажмите Q для выхода\n")
        
        score = 0
        
        try:
            for round_num in range(1, self.rounds + 1):
                # Выбираем случайную позу
                target_pose = random.choice(self.labels)
                
                # Играем раунд
                success, quit_game = self.play_round(round_num, target_pose)
                
                if quit_game:
                    print("\nИгра прервана пользователем")
                    break
                
                if success:
                    score += 1
                    print(f"Раунд {round_num}: УСПЕХ! (Счет: {score}/{round_num})")
                else:
                    print(f"Раунд {round_num}: Время вышло (Счет: {score}/{round_num})")
                
                # Пауза между раундами (кроме последнего)
                if round_num < self.rounds and not quit_game:
                    print("Готовьтесь к следующему раунду...")
                    time.sleep(2)
            
            # Показываем финальный результат
            self.show_final_score(score)
            
        except KeyboardInterrupt:
            print("\nИгра прервана")
        finally:
            self.cleanup()
    
    def show_final_score(self, score):
        """Показываем финальный счет"""
        print(f"\n=== ИГРА ЗАВЕРШЕНА ===")
        print(f"Итоговый счет: {score}/{self.rounds}")
        
        # Показываем результат на экране
        for _ in range(90):  # 3 секунды при 30 FPS
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2BGR)
                
                # Рисуем финальный результат
                self.draw_text(image, "GAME OVER", (180, 200), (255, 255, 255), 2.0, 4)
                self.draw_text(image, f"Final Score: {score}/{self.rounds}", (120, 280), (255, 255, 0), 1.5, 3)
                
                # Оценка результата
                if score == self.rounds:
                    self.draw_text(image, "PERFECT!", (160, 350), (0, 255, 0), 1.5, 3)
                elif score >= self.rounds * 0.7:
                    self.draw_text(image, "GOOD JOB!", (140, 350), (0, 255, 255), 1.5, 3)
                else:
                    self.draw_text(image, "KEEP TRYING!", (100, 350), (255, 165, 0), 1.5, 3)
                
                cv2.imshow('Pose Game - Press Q to quit', image)
                if cv2.waitKey(33) & 0xFF == ord('q'):
                    break
    
    def cleanup(self):
        """Освобождаем ресурсы"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Ресурсы освобождены")

def main():
    """Основная функция"""
    try:
        game = PoseGame()
        game.play()
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        print("Убедитесь, что файл 'pose_classifier.pkl' находится в той же папке")
    except RuntimeError as e:
        print(f"Ошибка: {e}")
        print("Проверьте подключение камеры")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")

if __name__ == "__main__":
    main()
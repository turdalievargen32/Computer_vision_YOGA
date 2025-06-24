import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import argparse

# ==== Параметры ====
MODEL_PATH = 'saved_model/yoga_model.h5'
IMG_SIZE = (224, 224)
DATA_DIR = os.path.expanduser('~/Desktop/Asyl_project/DATASET/TRAIN')

# ==== Получаем список классов ====
CLASSES = sorted(os.listdir(DATA_DIR))

# ==== Функция предсказания ====
def predict_pose(img_path):
    # Загружаем изображение
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)  # нормализация для MobileNetV2
    img_array = np.expand_dims(img_array, axis=0)

    # Загружаем модель
    model = load_model(MODEL_PATH)

    # Предсказание
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = CLASSES[predicted_index]
    confidence = predictions[0][predicted_index]

    print(f"\n🧘 Предсказанная поза: **{predicted_class}**")
    print(f"🔍 Уверенность: {confidence * 100:.2f}%\n")

# ==== Точка входа ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Путь к изображению для предсказания')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print("❌ Изображение не найдено. Проверь путь.")
    else:
        predict_pose(args.image)

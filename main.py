import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ─── Параметры ──────────────────────────────────────────────
DATA_DIR   = "/home/user/Desktop/Asyl_project/DATASET/TRAIN"   # путь к датасету
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 5                              # увеличь до 10–15 при желании
# ────────────────────────────────────────────────────────────

# 1. Получаем список классов
classes = sorted(os.listdir(DATA_DIR))
print(f"Найдено {len(classes)} классов: {classes[:8]} ...")

# 2. Генераторы данных с валидацией 20 %
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 3. Строим модель на базе MobileNetV2
base = MobileNetV2(weights='imagenet', include_top=False,
                   input_shape=IMG_SIZE + (3,))
base.trainable = False        # замораживаем базу (можно разморозить позже)

x   = GlobalAveragePooling2D()(base.output)
out = Dense(len(classes), activation='softmax')(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer=Adam(1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 4. Обучаем
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen
)

# 5. Сохраняем
os.makedirs("saved_model", exist_ok=True)
model_path = "saved_model/yoga_model.h5"
model.save(model_path)
print(f"✅ Модель сохранена: {model_path}")

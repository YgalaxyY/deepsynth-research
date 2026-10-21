import os
import shutil
import pandas as pd
from pathlib import Path
import random
import re

# Пути к папкам (используем абсолютные пути)
BASE_DIR = Path(__file__).parent.parent
SOURCE_DIR = BASE_DIR / "dataset_source" / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2"
TARGET_DIR = BASE_DIR / "working_data" / "original"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# Сколько видео берем для эксперимента (чтобы ноутбук не взорвался)
NUM_SAMPLES_PER_CLASS = 200
random.seed(42) 

def get_videos(category_name):
    # Ищем все .mp4 файлы внутри конкретной категории (например, RealVideo-RealAudio)
    category_path = SOURCE_DIR / category_name
    print(f"Проверяю путь: {category_path}")
    print(f"Путь существует: {category_path.exists()}")
    
    videos = sorted(category_path.rglob("*.mp4"))
    print(f"Найдено видео в {category_name}: {len(videos)}")
    return videos

print("Ищем видео...")
# Genuine samples are RealVideo-RealAudio. FakeVideo-RealAudio is manipulated video.
real_videos = get_videos("RealVideo-RealAudio")
fake_videos = get_videos("FakeVideo-FakeAudio")

print(f"Найдено подлинных видео и аудио: {len(real_videos)}")
print(f"Найдено видео с фейковым видео-фейковым аудио: {len(fake_videos)}")

# Случайно выбираем по 200 штук
selected_real = random.sample(real_videos, min(NUM_SAMPLES_PER_CLASS, len(real_videos)))
selected_fake = random.sample(fake_videos, min(NUM_SAMPLES_PER_CLASS, len(fake_videos)))

metadata = []

# Функция копирования
def process_files(file_list, label):
    for i, file_path in enumerate(file_list):
        new_name = f"{'real' if label == 0 else 'fake'}_{i:03d}.mp4"
        dest_path = TARGET_DIR / new_name
        shutil.copy(file_path, dest_path) # Копируем в рабочую папку
        subject = next((part for part in reversed(file_path.parts[:-1]) if re.fullmatch(r"id\d+", part)), None)
        if not subject:
            raise ValueError(f"No subject directory in {file_path}")
        metadata.append({"filename": new_name, "label": label, "source_group": subject})

print("Копируем файлы...")
process_files(selected_real, 0) # Метка 0 - оригинал
process_files(selected_fake, 1) # Метка 1 - фейк

# Сохраняем разметку
df = pd.DataFrame(metadata)
metadata_path = BASE_DIR / "working_data" / "metadata.csv"
df.to_csv(metadata_path, index=False)
print("Готово! Создан metadata.csv")
print(f"Всего обработано файлов: {len(metadata)}")
print(f"Файл сохранен в: {metadata_path}")

import cv2
import subprocess
import shutil
import torch
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

# Используем абсолютные пути
BASE_DIR = Path(__file__).parent.parent

# 1. Загружаем нейросеть для видео (EfficientNet-B4)
print("Загрузка EfficientNet-B4...")
weights = models.EfficientNet_B4_Weights.DEFAULT
model_vis = models.efficientnet_b4(weights=weights)
model_vis.classifier = torch.nn.Identity() # Убираем слой классификации, чтобы получить вектор из 1792 чисел
model_vis.eval() # Режим сканирования

# Трансформации картинки для нейросети
preprocess = weights.transforms()


# Папки
df = pd.read_csv(BASE_DIR / "working_data" / "metadata.csv")
FEAT_DIR = BASE_DIR / "features"
FEAT_DIR.mkdir(exist_ok=True)

def get_iqa_score(frame):
    # Реализация модуля IQA из Главы 2.3 (Дисперсия Лапласа)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Нормализуем: считаем, что значение 500 и выше - это q=1.0 (идеально), а всё что ниже стремится к 0
    q = min(1.0, blur_score / 500.0)
    return max(0.01, q) # q не должно быть полным нулем

def extract_audio_features(video_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required for audio decoding.")
    result = subprocess.run([ffmpeg, "-v", "error", "-i", str(video_path), "-vn",
                             "-ac", "1", "-ar", "16000", "-t", "2", "-f", "f32le", "pipe:1"],
                            capture_output=True, check=True)
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    if samples.size < 400:
        raise ValueError(f"Missing or insufficient audio: {Path(video_path).name}")
    return librosa.feature.mfcc(y=samples, sr=16000, n_mfcc=128).mean(axis=1)


def process_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret: raise ValueError(f"Cannot decode first frame: {Path(video_path).name}")

    # 1. Считаем качество q (Модуль IQA)
    q_score = get_iqa_score(frame)
    
    # 2. Визуальные признаки
    # В реальной системе тут MediaPipe вырезает лицо. Для прототипа берем центр кадра.
    img_t = preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0)
    with torch.no_grad():
        v_feat = model_vis(img_t).squeeze().numpy() # 1792 числа
        
    # 3. Аудио признаки
    a_feat = extract_audio_features(video_path) # 128 чисел
    
    return q_score, v_feat, a_feat

# Прогоняем все качества (hq, mq, lq)
for q_lvl in ['hq', 'mq', 'lq']:
    print(f"Извлечение признаков для {q_lvl.upper()}...")
    feat_list = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        vid_path = BASE_DIR / "working_data" / q_lvl / row['filename']
        q, v_feat, a_feat = process_video(vid_path)
        
        if q is not None:
            feat_list.append({
                'filename': row['filename'],
                'label': row['label'],
                'source_group': row['source_group'],
                'q_score': q,
                'v_feat': v_feat,
                'a_feat': a_feat
            })
            
    # Сохраняем в файл .npy (быстрый бинарный формат)
    np.save(FEAT_DIR / f"features_{q_lvl}.npy", feat_list)
    print(f"Сохранено {len(feat_list)} признаков для {q_lvl}")

print("Извлечение признаков завершено!")

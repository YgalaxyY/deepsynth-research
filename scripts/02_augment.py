import os
import subprocess
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Используем абсолютные пути
BASE_DIR = Path(__file__).parent.parent
df = pd.read_csv(BASE_DIR / "working_data" / "metadata.csv")
ORIGINAL_DIR = BASE_DIR / "working_data" / "original"

# Создаем папки для разного качества
QUALITIES = {
    "hq": {"crf": "18", "scale": None},      # Идеальное (Baseline)
    "mq": {"crf": "28", "scale": "480"},     # Имитация Telegram
    "lq": {"crf": "40", "scale": "360"}      # Имитация WhatsApp
}

for q_name, params in QUALITIES.items():
    out_dir = BASE_DIR / "working_data" / q_name
    out_dir.mkdir(exist_ok=True)
    print(f"\nГенерация качества: {q_name.upper()}")
    
    for filename in tqdm(df['filename']):
        in_file = ORIGINAL_DIR / filename
        out_file = out_dir / filename
        
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("Install FFmpeg and add it to PATH before augmentation.")
        cmd = [ffmpeg, "-y", "-i", str(in_file), "-vcodec", "libx264", "-crf", params["crf"]]
        if params['scale']:
            cmd += ["-vf", f"scale=-2:{params['scale']}"]
        cmd += ["-acodec", "aac", "-b:a", "64k", str(out_file), "-loglevel", "error"]
        subprocess.run(cmd, check=True)

print("\nАугментация завершена!")

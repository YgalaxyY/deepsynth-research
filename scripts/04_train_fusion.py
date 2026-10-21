import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import json
from protocol import split_subjects
from pathlib import Path

# Используем абсолютные пути
BASE_DIR = Path(__file__).parent.parent

torch.manual_seed(42)
np.random.seed(42)

# Quality-aware fusion architecture
class ConfidenceAwareFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.vis_mlp = nn.Sequential(nn.Linear(1792, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
        self.aud_mlp = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        
        self.k = 10.0 # Коэффициент крутизны
        self.q0 = 0.4 # Порог перегиба (когда перестаем верить видео)

    def forward(self, v_feat, a_feat, q_score):
        p_v = self.vis_mlp(v_feat)
        p_a = self.aud_mlp(a_feat)
        
        # Динамическое слияние!
        w_v = 1.0 / (1.0 + torch.exp(-self.k * (q_score - self.q0)))
        w_a = 1.0 - w_v
        return (w_v * p_v) + (w_a * p_a)

# 2. Загрузка данных (Обучаем на высоком качестве HQ)
print("Загрузка признаков HQ...")
data = np.load(BASE_DIR / "features" / "features_hq.npy", allow_pickle=True)
v_feats = torch.tensor(np.array([d['v_feat'] for d in data]), dtype=torch.float32)
a_feats = torch.tensor(np.array([d['a_feat'] for d in data]), dtype=torch.float32)
q_scores = torch.tensor(np.array([d['q_score'] for d in data]), dtype=torch.float32).unsqueeze(1)
labels = torch.tensor(np.array([d['label'] for d in data]), dtype=torch.float32).unsqueeze(1)

print(f"Загружено {len(data)} образцов")

# Разбиваем на Train (Обучение) и Test (Проверка)
split = split_subjects(list(data), seed=42)
train_ids, test_ids = set(split['train']), set(split['test'])
train_idx = [i for i, row in enumerate(data) if row['filename'] in train_ids]
test_idx = [i for i, row in enumerate(data) if row['filename'] in test_ids]
v_train, v_test = v_feats[train_idx], v_feats[test_idx]
a_train, a_test = a_feats[train_idx], a_feats[test_idx]
q_train, q_test = q_scores[train_idx], q_scores[test_idx]
y_train, y_test = labels[train_idx], labels[test_idx]
(BASE_DIR/'scripts/split.json').write_text(json.dumps(split, indent=2), encoding='utf-8')

print(f"Размер обучающей выборки: {len(v_train)}")
print(f"Размер тестовой выборки: {len(v_test)}")

# 3. Цикл обучения
model = ConfidenceAwareFusion()
criterion = nn.BCELoss() # Ошибка бинарной классификации
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Начинаем обучение нашего адаптивного слоя...")
for epoch in range(30): # 30 эпох
    model.train()
    optimizer.zero_grad()
    
    outputs = model(v_train, a_train, q_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Эпоха {epoch+1}/30, Ошибка: {loss.item():.4f}")

# Сохраняем готовую модель
model_path = BASE_DIR / "scripts" / "fusion_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Модель обучена и сохранена в {model_path}")

# Проверка на тестовых данных
model.eval()
with torch.no_grad():
    test_outputs = model(v_test, a_test, q_test)
    test_loss = criterion(test_outputs, y_test)
    print(f"Тестовая ошибка: {test_loss.item():.4f}")

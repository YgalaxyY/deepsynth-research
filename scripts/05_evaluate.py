import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from pathlib import Path

# Используем абсолютные пути
BASE_DIR = Path(__file__).parent.parent

# Загружаем класс модели
class ConfidenceAwareFusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vis_mlp = torch.nn.Sequential(torch.nn.Linear(1792, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1), torch.nn.Sigmoid())
        self.aud_mlp = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1), torch.nn.Sigmoid())
        self.k = 10.0; self.q0 = 0.4
    def forward(self, v_feat, a_feat, q_score):
        p_v = self.vis_mlp(v_feat)
        p_a = self.aud_mlp(a_feat)
        w_v = 1.0 / (1.0 + torch.exp(-self.k * (q_score - self.q0)))
        w_a = 1.0 - w_v
        return (w_v * p_v) + (w_a * p_a), p_v # Возвращаем еще и предсказание только видео (Baseline)

# Загружаем обученную модель
model = ConfidenceAwareFusion()
model_path = BASE_DIR / "scripts" / "fusion_model.pth"
model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
model.eval()

test_ids = set(json.loads((BASE_DIR/'scripts/split.json').read_text(encoding='utf-8'))['test'])

results_auc = {'Baseline (Только Видео)':[], 'Proposed (Наша Адаптивная)': []}
crf_levels =['hq (CRF 18)', 'mq (CRF 28)', 'lq (CRF 40)']

print("Evaluation on held-out subjects only. Historical draft metrics are not reused.")

for q_lvl in ['hq', 'mq', 'lq']:
    print(f"\nОбработка {q_lvl.upper()}...")
    data = np.load(BASE_DIR / "features" / f"features_{q_lvl}.npy", allow_pickle=True)
    data = [row for row in data if row['filename'] in test_ids]
    if {row['filename'] for row in data} != test_ids:
        raise ValueError('A quality level is missing held-out samples; refusing a biased comparison.')
    v_t = torch.tensor(np.array([d['v_feat'] for d in data]), dtype=torch.float32)
    a_t = torch.tensor(np.array([d['a_feat'] for d in data]), dtype=torch.float32)
    q_t = torch.tensor(np.array([d['q_score'] for d in data]), dtype=torch.float32).unsqueeze(1)
    y_true = np.array([d['label'] for d in data])
    
    with torch.no_grad():
        y_pred_ours, y_pred_baseline = model(v_t, a_t, q_t)
    
    auc_ours = roc_auc_score(y_true, y_pred_ours.numpy())
    auc_base = roc_auc_score(y_true, y_pred_baseline.numpy())
    
    results_auc['Proposed (Наша Адаптивная)'].append(auc_ours)
    results_auc['Baseline (Только Видео)'].append(auc_base)
    print(f"Качество {q_lvl.upper()} -> Наша AUC: {auc_ours:.3f} | Baseline AUC: {auc_base:.3f}")

# Рисуем красивый график для диплома (Рисунок 6)
plt.figure(figsize=(10, 6))
plt.plot(crf_levels, results_auc['Baseline (Только Видео)'], marker='o', linestyle='--', color='red', label='Visual Only (Baseline)', linewidth=2)
plt.plot(crf_levels, results_auc['Proposed (Наша Адаптивная)'], marker='s', linestyle='-', color='blue', label='Confidence-aware Fusion (Наш)', linewidth=2)

plt.title('Деградация точности моделей при сжатии (Рисунок 6)', fontsize=14, fontweight='bold')
plt.xlabel('Уровень сжатия (компрессия)', fontsize=12)
plt.ylabel('Точность ROC-AUC', fontsize=12)
plt.ylim([0.4, 1.05])
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# Добавляем значения точек на график
for i, (baseline, proposed) in enumerate(zip(results_auc['Baseline (Только Видео)'], results_auc['Proposed (Наша Адаптивная)'])):
    plt.text(i, baseline + 0.02, f'{baseline:.3f}', ha='center', va='bottom', fontsize=10, color='red')
    plt.text(i, proposed + 0.02, f'{proposed:.3f}', ha='center', va='bottom', fontsize=10, color='blue')

plt.tight_layout()
chart_path = BASE_DIR / "scripts" / "result_chart.png"
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
print(f"\nГрафик сохранен как {chart_path}!")

# Выводим итоговые результаты
print("\n=== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ===")
print(f"Baseline (Только Видео): {results_auc['Baseline (Только Видео)']}")
print(f"Proposed (Наша Адаптивная): {results_auc['Proposed (Наша Адаптивная)']}")

# Вычисляем улучшение на низком качестве
improvement_lq = results_auc['Proposed (Наша Адаптивная)'][2] - results_auc['Baseline (Только Видео)'][2]
print(f"\nУлучшение на низком качестве (LQ): {improvement_lq:.3f} ({improvement_lq/results_auc['Baseline (Только Видео)'][2]*100:.1f}%)")

# script_4_2.py — Обучение модели поиска ОКПД2 код (исправленная версия)
# Алгоритм: TF-IDF + KNN (k=5, cosine distance)

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import pickle
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================================================
# 1. НАСТРОЙКА ВЕСОВ И ПРЕПРОЦЕССИНГ
# =========================================================
# Слова, которые нужно УСИЛИТЬ (аббревиатуры, ключевые термины)
HIGH_WEIGHT_WORDS = {'цкс', 'арм', 'рс', 'авм', 'ппк', 'жк', 'монитор', 'контроллер', 'модуль', 'плата', 'блок'}

# Слова, которые нужно ИГНОРИРОВАТЬ (предлоги, союзы, общие слова)
LOW_WEIGHT_WORDS = {
    'для', 'на', 'в', 'по', 'из', 'и', 'а', 'но', 'или', 'это', 'как', 'при', 'без',
    'система', 'устройство', 'типа', 'модель', 'версия', 'комплект', 'набор', 'продукция'
}

def preprocess_and_boost(text, boost_factor=3):
    """
    Очистка текста + усиление ключевых слов через повторение.
    Повторение слова увеличивает его TF в TF-IDF.
    """
    if not isinstance(text, str):
        return ""
    # Убираем пунктуацию, приводим к нижнему регистру
    text = re.sub(r'[^\w\s]', ' ', text).lower().strip()
    words = text.split()
    boosted = []
    for w in words:
        if w in HIGH_WEIGHT_WORDS:
            # Усиливаем вес через повторение токена
            boosted.extend([w] * boost_factor)
        elif w not in LOW_WEIGHT_WORDS:
            # Оставляем только значимые слова
            boosted.append(w)
    return ' '.join(boosted)


# =========================================================
# 2. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# =========================================================
logging.info("Загрузка очищенных данных...")
registry_df = pd.read_csv('registry_cleaned.csv')

# Применяем препроцессинг ко всему реестру
registry_df['processed_text'] = registry_df['Наименование РЭП'].apply(preprocess_and_boost)

logging.info(f"Всего записей: {len(registry_df)}")
logging.info(f"Уникальных ОКПД2 код: {registry_df['ОКПД2 код'].nunique()}")


# =========================================================
# 3. ВЕКТОРИЗАЦИЯ (fit на ВСЁМ датасете для поиска)
# =========================================================
# Для задачи ПОИСКА похожих векторизатор должен знать ВСЕ слова из базы
# Поэтому fit делаем на полном реестре, а не на train-выборке

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,           # Больше пространства для терминов
    ngram_range=(1, 1),          # Только униграммы (биграммы часто шумят в коротких текстах)
    min_df=1,                    # Берем все слова, встречающиеся хотя бы 1 раз
    max_df=0.95,                 # Не отсекаем частые, если они не в 95% документов
    lowercase=False,             # Текст уже в нижнем регистре после preprocess
    analyzer='word',             # 🔑 Ключевое: работаем со словами, а не символами
    token_pattern=r'(?u)\b\w{2,}\b',  # Слова от 2 символов (убирает одиночные буквы, но оставляет 'рс', 'цкс')
    sublinear_tf=True            # Сглаживает частоты: 1 + log(TF)
)

logging.info("\n1. Векторизация полного реестра...")
X_all = tfidf_vectorizer.fit_transform(registry_df['processed_text'])

logging.info(f"   - Размер матрицы: {X_all.shape}")
logging.info(f"   - Кол-во признаков: {X_all.shape[1]}")

# 🔍 Проверка: попали ли ключевые слова в словарь
logging.info("\n🔍 Проверка словаря векторизатора:")
for w in ['цкс', 'монитор', 'рдв', 'rdw2401h', 'арм', 'рс']:
    status = "✅ В словаре" if w in tfidf_vectorizer.vocabulary_ else "❌ Отсутствует (будет проигнорирована)"
    logging.info(f"   '{w}': {status}")


# =========================================================
# 4. ОБУЧЕНИЕ KNN НА ВСЕЙ БАЗЕ
# =========================================================
logging.info("\n2. Обучение KNN на полном реестре...")
knn_model = NearestNeighbors(
    n_neighbors=5,
    metric='cosine',
    n_jobs=-1
)
knn_model.fit(X_all)
logging.info(f"   - Модель обучена на {len(registry_df)} примерах")


# =========================================================
# 5. СОХРАНЕНИЕ МОДЕЛИ (с двумя ключами для совместимости!)
# =========================================================
# 🚀 Оптимизация: считаем частоты один раз при обучении
okpd2_freq_dict = registry_df['ОКПД2 код'].value_counts().to_dict()

model_data = {
    'vectorizer': tfidf_vectorizer,
    'knn_model': knn_model,
    'registry_df': registry_df,
    'train_df': registry_df,              # для обратной совместимости
    'okpd2_frequencies': okpd2_freq_dict, # ✅ Словарь {ОКПД2 код: частота}
    'valid_okpd2': set(registry_df['ОКПД2 код'].unique())
}

with open('okpd_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print("✓ Модель сохранена в okpd_model.pkl")


# =========================================================
# 6. ОЦЕНКА ТОЧНОСТИ (опционально, на случайной подвыборке)
# =========================================================
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logging.info("\n3. Оценка точности на случайной подвыборке...")
# Для оценки берём случайную выборку, но векторизатор уже обучен на всём
sample_idx = np.random.choice(len(registry_df), size=min(1000, len(registry_df)), replace=False)
sample_df = registry_df.iloc[sample_idx]
X_sample = tfidf_vectorizer.transform(sample_df['processed_text'])

distances, indices = knn_model.kneighbors(X_sample)

# Голосование ближайших соседей для классификации
y_pred = []
for neighbors in indices:
    okpd2_neighbors = registry_df.iloc[neighbors]['ОКПД2 код'].tolist()
    y_pred.append(Counter(okpd2_neighbors).most_common(1)[0][0])

y_true = sample_df['ОКПД2 код'].values
acc = accuracy_score(y_true, y_pred)
logging.info(f"✅ Точность (accuracy@1) на подвыборке: {acc:.2%}")


# =========================================================
# 7. ПРОВЕРКА СХОДСТВА НА ТЕСТОВЫХ ФРАЗАХ
# =========================================================
test_phrases = [
    "монитор для цкс монитор",
    "монитор rdw2401h",
    "цкс монитор а",
    "ЦКС Монитор-А",
    "АРМ оператора"
]

logging.info("\n📏 Матрица косинусного сходства:")
X_test = tfidf_vectorizer.transform([preprocess_and_boost(p) for p in test_phrases])
sim_matrix = cosine_similarity(X_test)

for i, p1 in enumerate(test_phrases):
    for j, p2 in enumerate(test_phrases):
        if i < j:
            logging.info(f"  '{p1}' ↔ '{p2}': {sim_matrix[i, j]:.3f}")


# =========================================================
# 8. ПРИМЕР ПОИСКА БЛИЖАЙШИХ СОСЕДЕЙ
# =========================================================
logging.info("\n🔍 Пример поиска ближайших соседей для запроса:")
query = "ЦКС Монитор-А"
query_processed = preprocess_and_boost(query)
X_query = tfidf_vectorizer.transform([query_processed])

distances, indices = knn_model.kneighbors(X_query, n_neighbors=3)

logging.info(f"Запрос: '{query}'")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
    original_name = registry_df.iloc[idx]['Наименование РЭП']
    okpd2 = registry_df.iloc[idx]['ОКПД2 код']
    similarity = 1 - dist  # Косинусное расстояние → сходство
    logging.info(f"  {i}. '{original_name}' [ОКПД2 код: {okpd2}] — сходство: {similarity:.3f}")

logging.info("\n🎉 Скрипт завершён успешно!")

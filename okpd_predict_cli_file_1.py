#!/usr/bin/env python3
"""
OKPD2 CLI Predictor для 1С
Запуск: python okpd_predict_cli.py "Товар1" "Товар2" -o результат.json
Вывод: JSON в файл + опционально в stdout
"""

import sys
import json
import pickle
import argparse
from pathlib import Path


def load_model(model_path: str):
    """Загружает модель из pickle-файла"""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_product(model_data, product_name: str, confidence_threshold: float = 0.7):
    vectorizer = model_data["vectorizer"]
    knn_model = model_data["knn_model"]

    # 🔑 Безопасное получение DataFrame (без оператора 'or')
    df = model_data.get("registry_df")
    if df is None:
        df = model_data.get("train_df")
    if df is None:
        raise ValueError("В модели отсутствует 'registry_df' или 'train_df'")

    # Векторизация и поиск 5 ближайших соседей
    x_input = vectorizer.transform([str(product_name)])
    distances, indices = knn_model.kneighbors(x_input, n_neighbors=5)

    similarities = 1 - distances[0]
    top_okpds = df.iloc[indices[0]]['ОКПД2 код'].tolist()
    top_products = df.iloc[indices[0]]['Наименование РЭП'].tolist()

    # Взвешенное голосование
    okpd_scores = {}
    okpd_counts = {}
    for okpd, sim in zip(top_okpds, similarities):
        okpd_scores[okpd] = okpd_scores.get(okpd, 0.0) + sim
        okpd_counts[okpd] = okpd_counts.get(okpd, 0) + 1

    sorted_okpds = sorted(okpd_scores.items(), key=lambda x: (-x[1], -okpd_counts[x[0]]))
    best_okpd, best_score = sorted_okpds[0]

    total_score = sum(okpd_scores.values())
    confidence = best_score / total_score if total_score > 0 else 0.0

    # Разрешение ничьих
    warning = None
    ties = [k for k, v in sorted_okpds if abs(v - best_score) < 1e-6]
    if len(ties) > 1:
        #freq_in_registry = df['ОКПД2'].value_counts()
        # 🚀 БЫСТРАЯ ВЕРСИЯ: берём частоты из модели, а не считаем заново
        freq_in_registry = model_data.get('okpd2_frequencies', {})
        best_okpd = max(ties, key=lambda x: freq_in_registry.get(x, 0))
        confidence *= 0.8
        warning = f"Найдено {len(ties)} равнозначных кодов. Выбран самый частый в реестре."

    best_match_idx = next((i for i, okpd in enumerate(top_okpds) if okpd == best_okpd), 0)
    similar_product = top_products[best_match_idx]
    best_similarity = similarities[best_match_idx]

    return {
        "product_name": product_name,
        "predicted_okpd2": best_okpd,
        "confidence": round(float(confidence), 4),
        "status": "SUCCESS" if confidence >= confidence_threshold else "LOW_CONFIDENCE",
        "similar_product": similar_product,
        "similarity": round(float(best_similarity), 4),
        "warning": warning,
        "top_5_matches": [
            {"okpd2": o, "product": p, "similarity": round(float(s), 4)}
            for o, p, s in zip(top_okpds, top_products, similarities)
        ]
    }

def main():
    parser = argparse.ArgumentParser(description="OKPD2 Predictor CLI для 1С")
    parser.add_argument("products", nargs="+", help="Названия товаров/услуг")
    parser.add_argument("-t", "--threshold", type=float, default=0.7, 
                       help="Порог уверенности (по умолчанию: 0.7)")
    parser.add_argument("-m", "--model", type=str, default="okpd_model.pkl",
                       help="Путь к файлу модели")
    parser.add_argument("-o", "--output", type=str, required=True,
                       help="Путь к файлу для вывода JSON (обязательно для 1С)")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Вывод логов в stderr")
    
    args = parser.parse_args()
    
    # Логирование
    def log(msg):
        if args.verbose:
            print(msg, file=sys.stderr)
    
    log(f"🔄 Загрузка модели: {args.model}")
    
    # Загрузка модели
    try:
        model_data = load_model(args.model)
        log(f"✓ Модель загружена")
    except FileNotFoundError:
        error = {"error": f"Модель не найдена: {args.model}"}
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(json.dumps(error, ensure_ascii=False))
        sys.exit(2)
    except Exception as e:
        error = {"error": f"Ошибка загрузки модели: {str(e)}"}
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(json.dumps(error, ensure_ascii=False))
        sys.exit(2)
    
    # Предсказания
    predictions = []
    for product in args.products:
        try:
            pred = predict_product(model_data, product, args.threshold)
            predictions.append(pred)
            log(f"✓ {product} → {pred['predicted_okpd2']} ({pred['confidence']:.2%})")
        except Exception as e:
            log(f"✗ Ошибка для '{product}': {str(e)}")
            predictions.append({
                "product_name": product,
                "status": "ERROR",
                "error": str(e)
            })
    
    # Формируем результат
    output = {
        "total": len(predictions),
        "predictions": predictions
    }
    
    # Записываем в файл (основной способ для 1С)
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(json.dumps(output, ensure_ascii=False, indent=2))
        log(f"✓ Результат записан в {args.output}")
    except Exception as e:
        print(f"Ошибка записи файла: {e}", file=sys.stderr)
        sys.exit(3)
    
    # Также выводим в stdout для отладки (1С может игнорировать)
    if args.verbose:
        print(json.dumps(output, ensure_ascii=False), file=sys.stderr)
    
    # Код возврата: 0 = все ОК, 1 = есть предупреждения, 2+ = ошибки
    has_errors = any(p.get("status") == "ERROR" for p in predictions)
    has_warnings = any(p.get("status") == "LOW_CONFIDENCE" for p in predictions)
    
    if has_errors:
        sys.exit(2)
    elif has_warnings:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

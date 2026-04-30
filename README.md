# ОКПД2 ML API

Система машинного обучения для определения кодов ОКПД2 радиоэлектронной продукции на основе названия номенклатуры.

## Основные компоненты

- **okpd_async_api.py** — асинхронный веб-сервис (FastAPI)
- **okpd_predict_cli_file_1.py** — CLI-утилита для интеграции с 1С
- **script_2_1.py** — подготовка и очистка данных
- **script_4_2.py** — обучение модели (TF-IDF + KNN)
- **okpd_model.pkl** — предобученная модель

## Быстрый старт

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск API-сервера
python okpd_async_api.py

# CLI-предсказание
python okpd_predict_cli_file_1.py "Товар1" "Товар2" -o result.json
```

## Структура данных

- `okpd-po-prom-produktsii.xlsx` — справочник валидных кодов ОКПД2
- `perechen-radio-elektronnoi-produktsii.xlsx` — реестр радиоэлектронной продукции

## Требования

См. `requirements.txt` (FastAPI, scikit-learn, pandas, numpy)

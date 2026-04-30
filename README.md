# ОКПД2 ML API

Система машинного обучения для определения кодов ОКПД2 радиоэлектронной продукции на основе названия номенклатуры.

## Основные компоненты

- **okpd_async_api.py** — асинхронный веб-сервис (FastAPI)
- **okpd_predict_cli_file_1.py** — CLI-утилита для интеграции с 1С
- **script_2_1.py** — подготовка и очистка данных
- **script_4_2.py** — обучение модели (TF-IDF + KNN)
- **okpd_model.pkl** — предобученная модель

## Быстрый старт

### Запуск API-сервера
```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск сервера
python okpd_async_api.py
```

### Интеграция с 1С (CLI-скрипт)

**1. Создание виртуального окружения:**
```bash
python -m venv venv_1c
```

**2. Активация окружения:**
- Windows: `venv_1c\Scripts\activate`
- Linux/Mac: `source venv_1c/bin/activate`

**3. Установка минимальных зависимостей для 1С:**
```bash
pip install -r requirements_1c.txt
```

**4. Тестирование работоспособности скрипта:**
```bash
python okpd_predict_cli_file_1.py "Ноутбук" "Планшет" -o test_result.json -v
```

Проверьте файл `test_result.json` — если содержит `predictions`, скрипт готов к работе из 1С.

### CLI-предсказание (общий случай)
```bash
python okpd_predict_cli_file_1.py "Товар1" "Товар2" -o result.json
```

## Структура данных

- `okpd-po-prom-produktsii.xlsx` — справочник валидных кодов ОКПД2
- `perechen-radio-elektronnoi-produktsii.xlsx` — реестр радиоэлектронной продукции

## Требования

См. `requirements.txt` (FastAPI, scikit-learn, pandas, numpy)

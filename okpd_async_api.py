# -*- coding: utf-8 -*-
"""
ОКПД2 ML API - Асинхронный веб-сервис для определения ОКПД2 радиоэлектронной продукции
Архитектура: FastAPI + asyncio + Apache (wsgi-compatible via Gunicorn)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import pickle
import json
import logging
from datetime import datetime
import asyncio
import uuid
from pathlib import Path
import traceback

# ============================================================================
# КОНФИГУРАЦИЯ И ЛОГИРОВАНИЕ
# ============================================================================

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('okpd_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Инициализация FastAPI приложения
app = FastAPI(
    title="ОКПД2 ML API",
    description="Асинхронный веб-сервис для определения ОКПД2 радиоэлектронной продукции",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS для Apache/Nginx
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# МОДЕЛИ ДАННЫХ
# ============================================================================

class PredictionRequest(BaseModel):
    """Запрос на предсказание"""
    product_name: str = Field(..., min_length=3, max_length=500, description="Название продукции")
    confidence_threshold: float = Field(default=0.3, ge=0, le=1, description="Минимальный порог уверенности")
    return_top_matches: bool = Field(default=True, description="Возвращать топ совпадений")
    return_top_n: int = Field(default=5, ge=1, le=10, description="Количество топ совпадений")


class PredictionResponse(BaseModel):
    """Ответ с предсказанием"""
    request_id: str
    status: str
    predicted_okpd2: str
    confidence: float
    similar_product: str
    top_5_matches: Optional[List[Dict]]
    warning: Optional[str] = None
    error: Optional[str] = None
    timestamp: str
    processing_time_ms: float


class BatchPredictionRequest(BaseModel):
    """Батч запрос"""
    product_names: List[str] = Field(..., min_items=1, max_items=100, description="Список названий продукции")
    confidence_threshold: float = Field(default=0.3, ge=0, le=1)


class HealthResponse(BaseModel):
    """Ответ проверки здоровья"""
    status: str
    model_loaded: bool
    timestamp: str


# ============================================================================
# ИНИЦИАЛИЗАЦИЯ МОДЕЛИ
# ============================================================================

model_data = None
vectorizer = None
knn_model = None
train_df = None
valid_okpd2 = None

def load_model():
    """Загружает обученную модель"""
    global model_data, vectorizer, knn_model, train_df, valid_okpd2
    try:
        with open('okpd_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        vectorizer = model_data['vectorizer']
        knn_model = model_data['knn_model']
        train_df = model_data['train_df']
        valid_okpd2 = model_data['valid_okpd2']
        logger.info(f"✓ Модель загружена: {len(train_df)} примеров, {len(valid_okpd2)} ОКПД2")
        return True
    except Exception as e:
        logger.error(f"✗ Ошибка загрузки модели: {e}")
        return False


def predict_okpd2_async(product_name: str, threshold: float = 0.3) -> dict:
    """
    Асинхронное предсказание ОКПД2
    """
    if vectorizer is None or knn_model is None or train_df is None:
        raise RuntimeError("Модель не загружена")
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Векторизация
        query_vector = vectorizer.transform([product_name])
        
        # Поиск ближайших соседей
        distances, indices = knn_model.kneighbors(query_vector, n_neighbors=5)
        similarities = 1 - distances[0]
        
        # Лучший результат
        best_idx = indices[0][0]
        best_similarity = float(similarities[0])
        best_okpd2 = str(train_df.iloc[best_idx]['ОКПД2 код'])
        best_product = str(train_df.iloc[best_idx]['Наименование РЭП'])
        
        # Топ совпадения
        top_matches = []
        for i in range(min(5, len(indices[0]))):
            idx = indices[0][i]
            top_matches.append({
                'okpd2': str(train_df.iloc[idx]['ОКПД2 код']),
                'product': str(train_df.iloc[idx]['Наименование РЭП']),
                'similarity': float(similarities[i])
            })
        
        # Определение статуса
        status = "SUCCESS"
        warning = None
        
        if best_okpd2 not in valid_okpd2:
            status = "INVALID_OKPD2"
            warning = f"ОКПД2 {best_okpd2} не в списке валидных кодов радиоэлектронной продукции"
        elif best_similarity < threshold:
            status = "LOW_CONFIDENCE"
            warning = f"Низкая уверенность ({best_similarity:.3f} < {threshold})"
        
        processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return {
            'predicted_okpd2': best_okpd2,
            'confidence': best_similarity,
            'similar_product': best_product,
            'top_5_matches': top_matches,
            'status': status,
            'warning': warning,
            'processing_time_ms': processing_time
        }
        
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}\n{traceback.format_exc()}")
        raise


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    logger.info("🚀 Запуск ОКПД2 ML API...")
    if load_model():
        logger.info("✓ Система готова к работе")
    else:
        logger.error("✗ Не удалось загрузить модель")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Проверка здоровья сервиса"""
    return HealthResponse(
        status="OK" if vectorizer is not None else "ERROR",
        model_loaded=vectorizer is not None,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Предсказание ОКПД2 для одного продукта
    
    Пример:
    ```json
    {
        "product_name": "Модуль светодиодный СМ-100",
        "confidence_threshold": 0.3,
        "return_top_matches": true
    }
    ```
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Новый запрос: {request.product_name}")
    
    try:
        result = predict_okpd2_async(
            request.product_name,
            request.confidence_threshold
        )
        
        logger.info(f"[{request_id}] Успех: {result['predicted_okpd2']} (confidence: {result['confidence']:.3f})")
        
        return PredictionResponse(
            request_id=request_id,
            status=result['status'],
            predicted_okpd2=result['predicted_okpd2'],
            confidence=result['confidence'],
            similar_product=result['similar_product'],
            top_5_matches=result['top_5_matches'] if request.return_top_matches else None,
            warning=result['warning'],
            error=None,
            timestamp=datetime.utcnow().isoformat(),
            processing_time_ms=result['processing_time_ms']
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Ошибка: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при предсказании: {str(e)}"
        )


@app.post("/batch_predict", tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Батч предсказания для нескольких продуктов
    
    Пример:
    ```json
    {
        "product_names": [
            "Светильник светодиодный MERCURY LED Ex 20",
            "Видеокамера МВК-1885",
            "Микросхема интегральная К5559"
        ],
        "confidence_threshold": 0.3
    }
    ```
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Батч запрос: {len(request.product_names)} продуктов")
    
    try:
        predictions = []
        
        for product_name in request.product_names:
            try:
                result = predict_okpd2_async(
                    product_name,
                    request.confidence_threshold
                )
                predictions.append({
                    'product_name': product_name,
                    'status': result['status'],
                    'predicted_okpd2': result['predicted_okpd2'],
                    'confidence': result['confidence'],
                    'warning': result['warning']
                })
            except Exception as e:
                predictions.append({
                    'product_name': product_name,
                    'status': 'ERROR',
                    'error': str(e)
                })
        
        logger.info(f"[{request_id}] Батч завершен: {len(predictions)} результатов")
        
        return {
            'request_id': request_id,
            'total_processed': len(predictions),
            'successful': sum(1 for p in predictions if p['status'] == 'SUCCESS'),
            'timestamp': datetime.utcnow().isoformat(),
            'predictions': predictions
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Ошибка батча: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при батч обработке: {str(e)}"
        )


@app.get("/stats", tags=["System"])
async def get_stats():
    """Получить статистику модели"""
    if train_df is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    return {
        'model_info': {
            'training_samples': len(train_df),
            'unique_okpd2_classes': len(valid_okpd2),
            'vectorizer_features': 1000,
            'algorithm': 'TF-IDF + KNN (k=5, cosine distance)'
        },
        'okpd2_distribution': train_df['ОКПД2'].value_counts().head(10).to_dict(),
        'top_okpd2_codes': list(train_df['ОКПД2'].value_counts().head(10).index),
        'timestamp': datetime.utcnow().isoformat()
    }


@app.get("/", tags=["System"])
async def root():
    """Информация о API"""
    return {
        'name': 'ОКПД2 ML API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'health': '/health',
            'predict_single': '/predict',
            'predict_batch': '/batch_predict',
            'stats': '/stats',
            'docs': '/docs',
            'redoc': '/redoc'
        },
        'timestamp': datetime.utcnow().isoformat()
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error(f"ValueError: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": f"Ошибка значения: {str(exc)}"}
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc):
    logger.error(f"RuntimeError: {exc}")
    return JSONResponse(
        status_code=503,
        content={"detail": f"Ошибка сервиса: {str(exc)}"}
    )


# ============================================================================
# WSGI ENTRY POINT ДЛЯ APACHE (через Gunicorn)
# ============================================================================

if __name__ == "__main__":
    # Для разработки:
    # python -m uvicorn okpd_async_api:app --host 0.0.0.0 --port 8000 --workers 4
    
    # Для production с Gunicorn:
    # gunicorn -w 4 -k uvicorn.workers.UvicornWorker \
    #          -b 0.0.0.0:8000 okpd_async_api:app
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)

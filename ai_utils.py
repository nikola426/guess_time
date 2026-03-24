import numpy as np
import json
import os
from collections import defaultdict

# ---------- 1. ИИ для обнаружения событий на видео ----------
class EventDetectionAI:
    """
    В реальном приложении здесь использовалась бы предобученная модель для детекции событий,
    например, VideoMAE или модель на основе аудио/визуальных признаков.
    Для демонстрации мы загружаем события из JSON-файла (заглушка).
    """
    def __init__(self, events_file='events.json'):
        self.events = []
        if os.path.exists(events_file):
            with open(events_file, 'r') as f:
                data = json.load(f)
                self.events = data.get('events', [])
        else:
            # Создаём тестовые события для видео static/video.mp4
            self.events = [
                {'video_id': 'video1', 'timestamp': 10.5, 'event_type': 'goal'},
                {'video_id': 'video1', 'timestamp': 45.2, 'event_type': 'shot'},
                {'video_id': 'video1', 'timestamp': 78.0, 'event_type': 'goal'},
            ]

    def detect_events(self, video_id):
        """Возвращает список событий для указанного видео"""
        return [e for e in self.events if e['video_id'] == video_id]


# ---------- 2. ИИ для корректировки очков (ScoringAI) ----------
class ScoringAI:
    """
    Простая нейросеть (однослойная) для корректировки очков в зависимости от
    того, сколько пользователей угадало данное событие.
    Обучается онлайн на каждом предсказании.
    """
    def __init__(self, learning_rate=0.01):
        self.weights = np.array([1.0])          # вес для количества угадавших
        self.lr = learning_rate
        self.event_popularity = defaultdict(int)  # сколько раз угадано событие

    def adjust_score(self, raw_score, event_id):
        """
        raw_score: исходные очки (например, на основе близости к событию)
        event_id: идентификатор события
        Возвращает скорректированные очки.
        """
        popularity = self.event_popularity.get(event_id, 0)
        # Линейная коррекция: чем больше угадавших, тем меньше очков
        correction = max(0, 1 - popularity * self.weights[0])
        adjusted = raw_score * correction
        return adjusted

    def update(self, raw_score, event_id, final_score):
        """
        Обучение на примере (предсказание - фактический скор после коррекции).
        Здесь мы просто сохраняем популярность события.
        В реальности можно использовать градиентный спуск для оптимизации весов.
        """
        self.event_popularity[event_id] += 1
        # Простой вариант: корректируем вес, чтобы приблизить adjusted к raw_score
        # (если много угадавших, вес увеличивается, сильнее снижая очки)
        popularity = self.event_popularity[event_id]
        target_correction = final_score / raw_score if raw_score != 0 else 1
        error = target_correction - (1 - popularity * self.weights[0])
        self.weights[0] += self.lr * error * popularity

# Глобальные экземпляры (инициализируются в app.py)
event_detector = EventDetectionAI()
scoring_ai = ScoringAI()
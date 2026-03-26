import os
from flask import Flask, render_template, request, jsonify
from models import db, User, Event, Prediction
from ai_utils import event_detector, scoring_ai
import numpy as np

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///game.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Создаём базу данных и тестового пользователя (при необходимости)
with app.app_context():
    db.create_all()
    if not User.query.filter_by(name='TestUser').first():
        user = User(name='TestUser')
        db.session.add(user)
        db.session.commit()

# Функция вычисления исходных очков на основе временной разницы
def compute_raw_score(predicted_time, actual_time, tolerance=2.0):
    """
    Чем ближе предсказание к реальному времени, тем больше очков.
    Максимум 100 очков при разнице 0, падает линейно до 0 при разнице >= tolerance.
    """
    diff = abs(predicted_time - actual_time)
    if diff >= tolerance:
        return 0
    return 100 * (1 - diff / tolerance)

@app.route('/')
def index():
    return render_template('index.html', video_file='video.mp4')

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    data = request.json
    user_name = data.get('user', 'TestUser')
    video_id = data.get('video_id', 'video1')
    predicted_time = float(data.get('time', 0))

    # Получаем пользователя
    user = User.query.filter_by(name=user_name).first()
    if not user:
        user = User(name=user_name)
        db.session.add(user)
        db.session.commit()

    # Получаем реальные события для этого видео (через ИИ)
    events = event_detector.detect_events(video_id)
    if not events:
        return jsonify({'error': 'No events found for this video'}), 404

    # Находим ближайшее событие к предсказанному времени
    closest_event = min(events, key=lambda e: abs(e['timestamp'] - predicted_time))
    actual_time = closest_event['timestamp']
    event_id = closest_event.get('id')  # в нашем прототипе id может отсутствовать, но для демо добавим

    # Если у события нет id в БД, создадим его (или найдём)
    event_obj = Event.query.filter_by(video_id=video_id, timestamp=actual_time).first()
    if not event_obj:
        event_obj = Event(video_id=video_id, timestamp=actual_time, event_type=closest_event['event_type'])
        db.session.add(event_obj)
        db.session.commit()
    event_id = event_obj.id

    # Вычисляем исходные очки
    raw_score = compute_raw_score(predicted_time, actual_time)

    # Корректируем очки с помощью ScoringAI
    final_score = scoring_ai.adjust_score(raw_score, event_id)

    # Сохраняем предсказание
    prediction = Prediction(
        user_id=user.id,
        video_id=video_id,
        predicted_time=predicted_time,
        actual_event_id=event_id,
        raw_score=raw_score,
        final_score=final_score
    )
    db.session.add(prediction)
    db.session.commit()

    # Обучаем ScoringAI на этом примере (обновляем популярность события)
    scoring_ai.update(raw_score, event_id, final_score)

    return jsonify({
        'status': 'success',
        'raw_score': raw_score,
        'final_score': final_score,
        'actual_time': actual_time,
        'event_type': closest_event['event_type']
    })

@app.route('/api/leaderboard', methods=['GET'])
def leaderboard():
    # Агрегируем очки по пользователям
    users = User.query.all()
    leaderboard_data = []
    for user in users:
        total_score = db.session.query(db.func.sum(Prediction.final_score)).filter(Prediction.user_id == user.id).scalar() or 0
        leaderboard_data.append({'name': user.name, 'total_score': total_score})
    leaderboard_data.sort(key=lambda x: x['total_score'], reverse=True)
    return jsonify(leaderboard_data)

if __name__ == '__main__':
    app.run(debug=True)
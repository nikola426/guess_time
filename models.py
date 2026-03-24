from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Event(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(50), nullable=False)   # идентификатор видео
    timestamp = db.Column(db.Float, nullable=False)       # время события в секундах
    event_type = db.Column(db.String(50))                 # 'goal', 'shot' и т.д.
    # ... другие поля

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    video_id = db.Column(db.String(50), nullable=False)
    predicted_time = db.Column(db.Float, nullable=False)
    actual_event_id = db.Column(db.Integer, db.ForeignKey('event.id'), nullable=True)
    raw_score = db.Column(db.Float, default=0.0)          # очки до корректировки ИИ
    final_score = db.Column(db.Float, default=0.0)        # очки после корректировки ИИ
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
from datetime import datetime
from app import db

class Anomaly(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    confidence = db.Column(db.Float, nullable=False) 
    path = db.Column(db.String(255), nullable=True)  # Path to saved image
    description = db.Column(db.String(255), nullable=True)
    
    def __repr__(self):
        return f'<Anomaly {self.id} at {self.timestamp}>'
    
    def to_dict(self):
        """Convert model to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'path': self.path,
            'description': self.description,
            'formatted_time': self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }

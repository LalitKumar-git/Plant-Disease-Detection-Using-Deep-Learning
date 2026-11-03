import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'predictions.db')
SCHEMA_PATH = os.path.join(BASE_DIR, 'schema.sql')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH, 'r') as f:
        conn.executescript(f.read())
    conn.close()

def insert_prediction(filename, image_path, plant_name, disease_name, confidence, status, symptoms, treatment, notes=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (filename, image_path, plant_name, disease_name, confidence, status, symptoms, treatment, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (filename, image_path, plant_name, disease_name, confidence, status, symptoms, treatment, notes))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id

def get_all_predictions(limit=100):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def get_db_stats():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as t FROM predictions")
    t = cursor.fetchone()['t']
    cursor.execute("SELECT COUNT(*) as d FROM predictions WHERE status='Diseased'")
    d = cursor.fetchone()['d']
    cursor.execute("SELECT COUNT(*) as h FROM predictions WHERE status='Healthy'")
    h = cursor.fetchone()['h']
    cursor.execute("SELECT AVG(confidence) as c FROM predictions")
    c = cursor.fetchone()['c'] or 0.0
    conn.close()
    return {"total_scans": t, "diseased_scans": d, "healthy_scans": h, "avg_confidence": round(c, 1)}

if __name__ == '__main__':
    init_db()
    print("Database initialized.")

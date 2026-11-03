import os
import time
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify

from database.db_setup import init_db, insert_prediction, get_all_predictions, get_db_stats
from models.model_loader import run_cnn_inference, get_weights_location_info
from models.disease_data import get_disease_info

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )
    
    

try:
    init_db()
except Exception as e:
    print(f"Database initialization failed: {e}")
    raise

@app.route('/')
def index():
    stats = get_db_stats()
    weights_info = get_weights_location_info()
    return render_template('index.html', stats=stats, weights_info=weights_info)

@app.route('/history')
def history():
    records = get_all_predictions()
    stats = get_db_stats()
    return render_template('history.html', records=records, stats=stats)


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file uploaded'}), 400

    file = request.files['image']

    # Step 3: Validate filename and extension
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Invalid file type. Upload PNG, JPG, JPEG or WEBP.'
        }), 400

    # Step 4: Check whether uploaded file is empty
    if file.stream.seek(0, os.SEEK_END) == 0:
        return jsonify({
            'success': False,
            'error': 'Uploaded file is empty'
        }), 400

    file.stream.seek(0)

    # Save uploaded image
    filename = secure_filename(f"{int(time.time())}_{file.filename}")
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    inference = run_cnn_inference(save_path)
    info = get_disease_info(inference['predicted_class'])

    record_id = insert_prediction(
        filename=file.filename,
        image_path=f"static/uploads/{filename}",
        plant_name=inference['crop'],
        disease_name=inference['disease'],
        confidence=inference['confidence'],
        status='Healthy' if inference['is_healthy'] else 'Diseased',
        symptoms=info.get('symptoms', 'N/A'),
        treatment=f"Chemical: {info.get('treatment_chemical', 'N/A')}. Organic: {info.get('treatment_organic', 'N/A')}"
    )

    return jsonify({
        'success': True,
        'record_id': record_id,
        'crop': inference['crop'],
        'disease': inference['disease'],
        'confidence': inference['confidence'],
        'is_healthy': inference['is_healthy'],
        'symptoms': info.get('symptoms'),
        'treatment': info.get('treatment_chemical'),
        'top_5': inference['top_5']
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from flask_cors import CORS
from waitress import serve

app = Flask(__name__)
CORS(app)


MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))


models = {
    'diabetes': {
        'model': joblib.load(os.path.join(MODELS_DIR, 'diabetes_model.pkl')),
        'scaler': joblib.load(os.path.join(MODELS_DIR, 'diabetes_scaler.pkl')),
        'expected_features': 8
    },
    'breast_cancer': {
        'model': joblib.load(os.path.join(MODELS_DIR, 'breast_cancer_model.pkl')),
        'scaler': joblib.load(os.path.join(MODELS_DIR, 'breast_cancer_scaler.pkl')),
        'expected_features': 30,
    },
    'heart_disease': {
        'model': joblib.load(os.path.join(MODELS_DIR, 'heart_disease_model.pkl')),
        'scaler': joblib.load(os.path.join(MODELS_DIR, 'heart_disease_scaler.pkl')),
        'expected_features': 13
    }
}



@app.route('/', methods=['GET'])
def home():
    return '''
        <h1>🏥 Disease Prediction API</h1>
        <p>API is running successfully. Use /predict/&lt;disease&gt; endpoint for predictions.</p>
        <h3>Available Models:</h3>
        <ul>
            <li><strong>Diabetes:</strong> /predict/diabetes (8 features)</li>
            <li><strong>Heart Disease:</strong> /predict/heart_disease (13 features)</li>
            <li><strong>Breast Cancer:</strong> /predict/breast_cancer (30 features)</li>
        </ul>
        <p><em>Send POST requests with JSON: {"features": [value1, value2, ...]}</em></p>
    '''



@app.route('/predict/<disease>', methods=['POST'])
def predict(disease):
    try:
        data = request.get_json()
        features = data.get('features')

        print("Requested disease:", disease)
        print("Available models:", list(models.keys()))

        if disease not in models:
            return jsonify({'error': f'Unknown disease model: {disease}'}), 400

        if not features:
            return jsonify({'error': 'No input features provided'}), 400


        expected_count = models[disease]['expected_features']
        if len(features) != expected_count:
            return jsonify({
                'error': f'Expected {expected_count} features for {disease}, got {len(features)}'
            }), 400


        features = np.array(features).reshape(1, -1)


        scaler = models[disease]['scaler']
        model = models[disease]['model']


        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)


        try:
            prediction_proba = model.predict_proba(scaled_features)
            confidence = float(max(prediction_proba[0])) * 100
        except:
            confidence = None

        return jsonify({
            'prediction': int(prediction[0]),
            'confidence': confidence,
            'disease': disease,
            'features_count': len(features[0])
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


#  Health Check Route
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(models.keys()),
        'total_models': len(models)
    })



@app.route('/info/<disease>', methods=['GET'])
def model_info(disease):
    if disease not in models:
        return jsonify({'error': f'Unknown disease model: {disease}'}), 400

    return jsonify({
        'disease': disease,
        'expected_features': models[disease]['expected_features'],
        'model_type': 'Random Forest Classifier',
        'status': 'loaded'
    })


if __name__ == '__main__':
    print(" Disease Prediction API Starting with Waitress...")
    print(" Models loaded:", list(models.keys()))
    print(" Server running on http://0.0.0.0:5000")
    serve(app, host='0.0.0.0', port=5000)
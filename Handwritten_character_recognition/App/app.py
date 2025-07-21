import os
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import traceback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
api_path = os.path.join(project_root, 'api')
sys.path.append(api_path)

from api.model_loader import load_models
from api.predict import predict_character

app = Flask(__name__)
CORS(app)

print("\n Loading EMNIST model...")
models = load_models()
emnist_model = models.get('emnist')


@app.route('/')
def index():

    frontend_path = os.path.join(project_root, 'frontend')
    template_path = os.path.join(frontend_path, 'index.html')
    app.template_folder = frontend_path
    return render_template("index.html") if os.path.exists(template_path) else "Frontend not found."


@app.route('/static/<path:filename>')
def static_files(filename):

    frontend_path = os.path.join(project_root, 'frontend')
    return send_from_directory(frontend_path, filename)


@app.route('/predict/character', methods=['POST'])
def predict_character_route():

    try:
        print(f" Received request: {request.method}")
        print(f" Files in request: {list(request.files.keys())}")
        print(f" Form data: {list(request.form.keys())}")
        print(f" Is JSON: {request.is_json}")

        image_data = None

        if 'file' in request.files and request.files['file']:
            image_data = request.files['file']
            print(f"File upload detected: {image_data.filename}, Size: {len(image_data.read())} bytes")
            image_data.seek(0)

        elif request.is_json:
            json_data = request.get_json()
            if 'image' in json_data:
                image_data = json_data['image']
                print("🖼 JSON image data detected")

        elif 'image' in request.form:
            image_data = request.form['image']
            print(" Form image data detected")

        elif request.data:
            try:
                import json
                data = json.loads(request.data)
                if 'image' in data:
                    image_data = data['image']
                    print(" Raw JSON image data detected")
            except:
                image_data = request.data.decode('utf-8')
                print(" Raw base64 data detected")

        if not image_data:
            error_msg = "No image data provided. Send as 'file' upload or 'image' in JSON/form data"
            print(f" {error_msg}")
            return jsonify({
                "error": error_msg,
                "success": False,
                "prediction": None,
                "confidence": 0.0,
                "top_predictions": [],
                "model": "EMNIST",
                "type": "character"
            }), 400

        if emnist_model is None:
            error_msg = "EMNIST model not loaded. Train one first."
            print(f" {error_msg}")
            return jsonify({
                "error": error_msg,
                "model": "EMNIST",
                "success": False,
                "prediction": None,
                "confidence": 0.0,
                "top_predictions": [],
                "type": "character"
            }), 500

        print(" Calling predict_character function...")


        prediction = predict_character(image_data, emnist_model, debug=True)

        print(f" Prediction result: {prediction}")

        return jsonify({
            "prediction": prediction['prediction'],
            "confidence": prediction['confidence'],
            "top_predictions": prediction['all_predictions'],
            "model": "EMNIST",
            "type": "character",
            "success": prediction['success'],
            "error": prediction.get('error'),
            "message": prediction.get('message')
        })

    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        print(f" Exception in predict_character_route: {error_msg}")
        print(f" Traceback: {traceback.format_exc()}")

        return jsonify({
            "error": error_msg,
            "success": False,
            "prediction": None,
            "confidence": 0.0,
            "top_predictions": [],
            "model": "EMNIST",
            "type": "character"
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "emnist_model_loaded": emnist_model is not None
    })


@app.errorhandler(404)
def not_found():
    return jsonify({
        "error": "Not Found",
        "message": "Available endpoints: /predict/character, /health"
    }), 404


@app.errorhandler(405)
def method_not_allowed():
    return jsonify({
        "error": "Method Not Allowed"
    }), 405


if __name__ == '__main__':
    print("\n Starting Flask App...")
    app.run(debug=True, host='0.0.0.0', port=8080)
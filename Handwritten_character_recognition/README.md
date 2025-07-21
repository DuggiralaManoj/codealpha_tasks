EMNIST Handwritten Character Recognition
A complete deep learning system for recognizing handwritten characters using Convolutional Neural Networks (CNNs) trained on the EMNIST dataset. Includes smart model merging to combine multiple trained models for improved accuracy.

 Project Overview
This project enables:

Training CNN models to recognize handwritten characters using EMNIST datasets.

Automatically searching, selecting, and merging multiple trained models to produce a more generalized and stable model.

Serving models through a simple Flask API.

 Key Features
 CNN-Based Character Recognition

 Support for EMNIST Dataset (ByClass variant)

 Automated Dataset Detection

 Training Monitoring with Plots

 Model Weight Merging (Averaging)

 Flask API for Remote Merging and Model Listing

 Auto-Selection of Best Models for Merging

 Components
1. Model Training (train_model.py)
Preprocesses CSV-based EMNIST datasets.

Builds a deep CNN model.

Applies dropout and batch normalization.

Saves trained models (.h5) and training history plots.

Automatically generates label mappings.

2. Smart Model Merging (mer_model.py)
Searches for trained models (.h5 / .keras).

Verifies model compatibility.

Merges multiple models by averaging their weights.

Produces a single, optimized model:
models/emnist_model.h5

Flask API Routes:

/find_models: List all available models.

/smart_merge: Trigger model merging via POST request.

3. REST API (Optional Deployment)
Lightweight Flask server.

Enables model management and merging remotely.
 
 
 Project Structure

├── train_model.py        # CNN model training script
├── mer_model.py          # Smart model merger & Flask API
├── requirements.txt      # Dependencies
├── README.md             # Project overview
├── models/               # Trained and merged models
├── datasets/             # EMNIST CSV datasets
├── training_history_*.png # Training plots
└── class_mapping_*.txt   # Class to character mapping


 Example Output
Trained Models:
models/emnist_byclass_model.h5

Merged Model:
models/emnist_model.h5

Training Plot:
training_history_byclass.png

Character Mapping:
class_mapping_byclass.txt

 Installation
pip install -r requirements.txt


 How to Use

➤ 1. Train New Model
python train_model.py

 2. Merge Existing Models
python mer_model.py

 3. Use Flask API (Optional)
from mer_model import add_smart_merge_routes
from flask import Flask

app = Flask(__name__)
add_smart_merge_routes(app)
app.run(port=5000)


Dependencies
TensorFlow

NumPy

Flask

OpenCV

SciPy

Scikit-learn

Pandas

Matplotlib

(See requirements.txt for exact versions.)

🏆 Goal
Improve handwritten character recognition performance by merging complementary CNN models, helping overcome single-model weaknesses and enhancing robustness.

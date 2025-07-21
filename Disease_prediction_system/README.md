
#  Disease Prediction System API

A Flask-based machine learning API that predicts the likelihood of **Diabetes**, **Heart Disease**, and **Breast Cancer** based on user input. The system uses pre-trained models and provides fast, accurate predictions via a RESTful API.

---

## ⚙ Technologies Used

- Python 3
- Flask + Waitress (for production)
- scikit-learn
- Pandas, NumPy
- Joblib (model saving/loading)
- KaggleHub (to download datasets)
- HTML, CSS, JavaScript (for web forms)
- Git + GitHub (version control)
- PyCharm (IDE)

---

##  How to Run the Project Locally

###  Prerequisites:
- Python 3.x installed
- Kaggle API key (`kaggle.json`) configured
- `pip install kagglehub`

### Steps:

1. **Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

2. **Create and activate a virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate   # For Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Train the models**

```bash
python training/model_training.py
```

5. **Run the Flask server**

```bash
python multidisease_api.py
```

6. **Open in browser**

Visit: `http://127.0.0.1:5000/`

---

##  How the Prediction API Works

### Endpoint Format

```
POST /predict/<disease>
```

- `<disease>` can be: `diabetes`, `heart_disease`, or `breast_cancer`
- Body format (JSON):

```json
{
  "features": [feature1, feature2, ..., featureN]
}
```

---

##  Example Requests and Responses

###  Example (breast_cancer)

**Request:**
```json
POST /predict/breast_cancer
{
  "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, ..., 0.07871]
}
```

**Response:**
```json
{
  "prediction": 1,
  "confidence": 96.0,
  "disease": "breast_cancer",
  "features_count": 30
}
```

###  Error Example:
```json
{
  "error": "Expected 31 features for breast_cancer, got 30"
}
```

---

##  Model Details

| Disease        | Dataset (Kaggle)                               | Algorithm                | Features |
|----------------|-----------------------------------------------|---------------------------|----------|
| Diabetes       | akshaydattatraykhare/diabetes-dataset         | Random Forest Classifier  | 8        |
| Heart Disease  | johnsmith88/heart-disease-dataset             | Random Forest Classifier  | 13       |
| Breast Cancer  | yasserh/breast-cancer-dataset                 | Random Forest Classifier  | 30       |

> All models use `StandardScaler` for preprocessing.

---

##  Folder Structure

```
Disease_Prediction_System/
├── training/
│   └── model_training.py         # Model training logic
├── models/                       # .pkl models and scalers
├── datasets/                     # Downloaded CSV datasets
├── static/
│   ├── main.js
│   └── stylesheet.css
├── templates/
│   ├── index.html
│   ├── diabetes_form.html
│   ├── heart_disease_form.html
│   └── breast_cancer_form.html
├── test.py                       # Local prediction test script
├── multidisease_api.py           # Flask prediction API
├── README.md                     # This file
```

---

##  Testing the API

### Using Python (`test.py`)
```bash
python test.py
```

### Using `curl`:
```bash
curl -X POST http://127.0.0.1:5000/predict/diabetes      -H "Content-Type: application/json"      -d '{"features": [6,148,72,35,0,33.6,0.627,50]}'
```

---

##  Future Enhancements

- Docker support for easy deployment
- Add more disease modules
- Frontend UI improvements (visualizations)
- Model explainability (e.g., SHAP)

---

##  Author

**Duggirala Manoj**  
GitHub: [@DuggiralaManoj](https://github.com/DuggiralaManoj)  
Email: dm9625178@gmail.com

---

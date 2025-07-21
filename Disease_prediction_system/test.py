import requests

url = "http://127.0.0.1:5000/predict/breast_cancer"

features = [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]

data = {"features": features}

response = requests.post(url, json=data)

if response.ok:
    print(" Prediction:", response.json())
else:
    print(" Error:", response.text)


# import requests
#
# url = "http://127.0.0.1:5000/predict/diabetes"
#
# features = [5, 116, 74, 0, 0, 25.6, 0.201, 30]
#
# data = {"features": features}
#
# response = requests.post(url, json=data)

# if response.ok:
#     print("Diabetes Prediction:", response.json())
# else:
#     print("Error:", response.text)




# import requests
#
# url = "http://127.0.0.1:5000/predict/heart_disease"
#
# features = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
#
# data = {"features": features}
#
# response = requests.post(url, json=data)
#
# if response.ok:
#     print("Heart Disease Prediction:", response.json())
# else:
#     print("Error:", response.text)


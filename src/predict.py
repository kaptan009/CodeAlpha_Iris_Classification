import joblib
import numpy as np

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

sample = np.array([[5.1,3.5,1.4,0.2]])
sample = scaler.transform(sample)

print(model.predict(sample))

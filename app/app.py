import streamlit as st
import numpy as np
import joblib

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Iris Classifier")

sl = st.slider("Sepal Length",4.0,8.0)
sw = st.slider("Sepal Width",2.0,4.5)
pl = st.slider("Petal Length",1.0,7.0)
pw = st.slider("Petal Width",0.1,2.5)

if st.button("Predict"):
    x = np.array([[sl,sw,pl,pw]])
    x = scaler.transform(x)
    pred = model.predict(x)
    st.write(pred[0])

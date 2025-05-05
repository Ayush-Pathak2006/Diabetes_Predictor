import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

diabetes_dataset = pd.read_csv(r'diabetes.csv')
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)

scaler= StandardScaler()

scaler.fit(X)
# Load the model
# model = pickle.load(open('diabetes_model.pkl', 'rb'))


# st.title('Diabetes Prediction App')

# # Input fields
# preg = st.number_input('Pregnancies')
# glucose = st.number_input('Glucose')
# bp = st.number_input('Blood Pressure')
# skin = st.number_input('Skin Thickness')
# insulin = st.number_input('Insulin')
# bmi = st.number_input('BMI')
# dpf = st.slider('Family History of Diabetes', 0.0, 2.5, 0.5)
# age = st.number_input('Age')

# if st.button('Predict'):
#     data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

model = pickle.load(open('diabetes_model.pkl', 'rb'))

# Optional: Load scaler if used
# scaler = pickle.load(open('scaler.pkl', 'rb'))

# Page configuration
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

# App title
st.title("🩺 Diabetes Prediction")
st.subheader("Enter patient details below:")

# Input form
with st.form("diabetes_form"):
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0.0, max_value=200.0, value=0.0)
    bp = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=150.0, value=0.0)
    skin = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=0.0)
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=0.0)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.0)
    age = st.number_input("Age", min_value=1, max_value=120, value=25)

    submitted = st.form_submit_button("Predict")

# Prediction
if submitted:
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    std_data = scaler.transform(input_data_reshaped)
    # data = np.array([[1, 103, 30, 38, 83, 43.3, 0.183, 33]])
    result = model.predict(std_data)
    if result[0]==1:
        st.success('This Person has high chance of Diabetes')
    else:
        st.success('This Person has low chance of Diabetes')
    #st.success( 'The patient has diabetes' if result[0] == 1 else 'The patient does not have diabetes')

st.markdown(
    """
    <hr style="margin-top: 50px; margin-bottom: 10px;">
    <div style='text-align: center;'>
        Made with ❤️ by <b>Ayush Pathak</b> and <b>Bhavya Soni</b>
    </div>
    """,
    unsafe_allow_html=True
)


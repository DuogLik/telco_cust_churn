import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import os

# Đặt cấu hình trang
st.set_page_config(page_title='Customer Churn Prediction', page_icon=':bar_chart:', layout='wide')

# Tạo thư mục để lưu lịch sử dự đoán
if not os.path.exists("prediction_history"):
    os.makedirs("prediction_history")

# Load các mô hình và dữ liệu test
@st.cache_resource
def load_models():
    logistic_regression_model = pickle.load(open("C:/Users/assus/PyCharmProject/Customers_Churn/MODELS/logistic_regression_model.pkl", "rb"))
    knn_model = pickle.load(open("C:/Users/assus/PyCharmProject/Customers_Churn/MODELS/knn_model.pkl", "rb"))
    random_forest_model = pickle.load(open("C:/Users/assus/PyCharmProject/Customers_Churn/MODELS/random_forest_model.pkl", "rb"))
    decision_tree_model = pickle.load(open("C:/Users/assus/PyCharmProject/Customers_Churn/MODELS/decision_tree_model.pkl", "rb"))
    xgboost_model = pickle.load(open("C:/Users/assus/PyCharmProject/Customers_Churn/MODELS/xgboost_model.pkl", "rb"))
    nn_model = load_model("C:/Users/assus/PyCharmProject/Customers_Churn/MODELS/neural_network_model.keras")
    cnn_model = load_model("C:/Users/assus/PyCharmProject/Customers_Churn/MODELS/cnn_model.keras")
    return {
        "Logistic Regression": logistic_regression_model,
        "KNN": knn_model,
        "Random Forest": random_forest_model,
        "Decision Tree": decision_tree_model,
        "XGBoost": xgboost_model,
        "Neural Network": nn_model,
        "CNN": cnn_model
    }

@st.cache_resource
def load_data():
    test_data = pd.read_csv("C:/Users/assus/PyCharmProject/Customers_Churn/data/test.csv")
    return test_data

models = load_models()
test_data = load_data()

# Hàm để dự đoán churn bằng các mô hình
def predict_churn(input_data, model_name):
    scaler = MinMaxScaler()
    scaler.fit(test_data.drop(columns=['churn']))
    input_data_scaled = scaler.transform(input_data)

    if model_name in ["Logistic Regression", "KNN", "Random Forest", "Decision Tree", "XGBoost"]:
        prediction = models[model_name].predict_proba(input_data_scaled)[:, 1]
    elif model_name in ["Neural Network", "CNN"]:
        if model_name == "CNN":
            input_data_scaled = input_data_scaled.reshape(input_data_scaled.shape[0], input_data_scaled.shape[1], 1)
        prediction = models[model_name].predict(input_data_scaled)
    return prediction

def save_prediction(prediction_data):
    history_file = "prediction_history/history.csv"
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
        history_df = pd.concat([history_df, prediction_data], ignore_index=True)
    else:
        history_df = prediction_data
    history_df.to_csv(history_file, index=False)

def load_prediction_history():
    history_file = "prediction_history/history.csv"
    if os.path.exists(history_file):
        return pd.read_csv(history_file)
    else:
        return pd.DataFrame()

def main():
    st.title('Telecom Customer Churn Prediction WEBAPP')
    st.write('This webapp will help you predict customer churn based on a Telecom Customer Churn Prediction.'
             ' You will predict the churn rate of customers in a telecom company using a stored model '
             ' based on Logistic Regression, KNN, Random Forest, Decision Tree, XGBoost, Neural Network, CNN.'
             ' To check the accuracy of the classifier, click on the Performance on Test Dataset button in the sidebar.'
             ' To predict, select the model you want to use from the dropdown box in the sidebar after choosing the user input data.')

    st.image("C:/Users/assus\PyCharmProject\Customers_Churn/maxresdefault.jpg", width=1100)
    st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .main .block-container {
        max-width: 80%;
        margin: auto;
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.title('MENU')
    task = st.sidebar.radio("Select Task", ["Predict", "Performance on Test Dataset", "Prediction History"])

    if task == "Predict":
        st.subheader('Input Customer Details')
        col1, col2 = st.columns(2)

        with col1:
            tenure = st.number_input("Tenure", value=int(test_data['tenure'].mean()))
            PhoneService = st.selectbox("Phone Service", ["Yes", "No", "No Phone Service"])
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox('Paperless Billing', ['Yes', 'No'])
            PaymentMethod = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            MonthlyCharges = st.number_input('Monthly Charges', value=float(test_data['monthly_charges'].mean()))
            TotalCharges = st.number_input('Total Charges', value=float(test_data['total_charges'].mean()))
            StreamingTV = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
            StreamingMovies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])

        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            SeniorCitizen = st.selectbox('Senior Citizen', ['Yes', 'No'])
            Partner = st.selectbox('Partner', ['Yes', 'No'])
            Dependents = st.selectbox('Dependents', ['Yes', 'No'])
            MultipleLines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
            InternetService = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
            OnlineSecurity = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
            OnlineBackup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
            DeviceProtection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
            TechSupport = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])

        input_data = pd.DataFrame({
            "tenure": [tenure],
            "monthly_charges": [MonthlyCharges],
            "total_charges": [TotalCharges],
            "senior_citizen": [1 if SeniorCitizen == "Yes" else 0],
            "phone_service_No": [1 if PhoneService == "No" else 0],
            "phone_service_Yes": [1 if PhoneService == "Yes" else 0],
            "contract_Month-to-month": [1 if Contract == "Month-to-month" else 0],
            "contract_One year": [1 if Contract == "One year" else 0],
            "contract_Two year": [1 if Contract == "Two year" else 0],
            "paperless_billing_No": [1 if PaperlessBilling == "No" else 0],
            "paperless_billing_Yes": [1 if PaperlessBilling == "Yes" else 0],
            "payment_method_Bank transfer (automatic)": [1 if PaymentMethod == "Bank transfer (automatic)" else 0],
            "payment_method_Credit card (automatic)": [1 if PaymentMethod == "Credit card (automatic)" else 0],
            "payment_method_Electronic check": [1 if PaymentMethod == "Electronic check" else 0],
            "payment_method_Mailed check": [1 if PaymentMethod == "Mailed check" else 0],
            "gender_Female": [1 if gender == "Female" else 0],
            "gender_Male": [1 if gender == "Male" else 0],
            "partner_No": [1 if Partner == "No" else 0],
            "partner_Yes": [1 if Partner == "Yes" else 0],
            "dependents_No": [1 if Dependents == "No" else 0],
            "dependents_Yes": [1 if Dependents == "Yes" else 0],
            "multiple_lines_No": [1 if MultipleLines == "No" else 0],
            "multiple_lines_No phone service": [1 if MultipleLines == "No phone service" else 0],
            "multiple_lines_Yes": [1 if MultipleLines == "Yes" else 0],
            "internet_service_DSL": [1 if InternetService == "DSL" else 0],
            "internet_service_Fiber optic": [1 if InternetService == "Fiber optic" else 0],
            "internet_service_No": [1 if InternetService == "No" else 0],
            "online_security_No": [1 if OnlineSecurity == "No" else 0],
            "online_security_No internet service": [1 if OnlineSecurity == "No internet service" else 0],
            "online_security_Yes": [1 if OnlineSecurity == "Yes" else 0],
            "online_backup_No": [1 if OnlineBackup == "No" else 0],
            "online_backup_No internet service": [1 if OnlineBackup == "No internet service" else 0],
            "online_backup_Yes": [1 if OnlineBackup == "Yes" else 0],
            "device_protection_No": [1 if DeviceProtection == "No" else 0],
            "device_protection_No internet service": [1 if DeviceProtection == "No internet service" else 0],
            "device_protection_Yes": [1 if DeviceProtection == "Yes" else 0],
            "tech_support_No": [1 if TechSupport == "No" else 0],
            "tech_support_No internet service": [1 if TechSupport == "No internet service" else 0],
            "tech_support_Yes": [1 if TechSupport == "Yes" else 0],
            "streaming_tv_No": [1 if StreamingTV == "No" else 0],
            "streaming_tv_No internet service": [1 if StreamingTV == "No internet service" else 0],
            "streaming_tv_Yes": [1 if StreamingTV == "Yes" else 0],
            "streaming_movies_No": [1 if StreamingMovies == "No" else 0],
            "streaming_movies_No internet service": [1 if StreamingMovies == "No internet service" else 0],
            "streaming_movies_Yes": [1 if StreamingMovies == "Yes" else 0]
        })

        if (PhoneService == " " or Contract == " " or PaperlessBilling == '' or PaymentMethod == '' or gender == '' or
            SeniorCitizen == '' or Partner == '' or Dependents == '' or MultipleLines == '' or InternetService == '' or
            OnlineSecurity == '' or OnlineBackup == '' or DeviceProtection == '' or TechSupport == '' or StreamingTV == '' or StreamingMovies == ''):
            st.warning("Please fill in all input fields to make a prediction.")
        else:
            model_name = st.sidebar.selectbox("Select Model", ["Logistic Regression", "KNN", "Random Forest", "Decision Tree", "XGBoost", "Neural Network", "CNN"])

            if st.sidebar.button("Predict"):
                prediction = predict_churn(input_data, model_name)
                prediction = prediction[0] * 100  # Convert to percentage
                st.subheader('Prediction Result')
                st.write("Churn Probability:", round(float(prediction), 2), "%")
                if prediction < 50:
                    st.success('NOT Churn')
                else:
                    st.error('CHURN')

                # Lưu kết quả dự đoán
                prediction_data = input_data.copy()
                prediction_data["churn_probability"] = round(float(prediction), 2)
                save_prediction(prediction_data)

    elif task == "Performance on Test Dataset":
        st.subheader("Performance on The Test Dataset (ROC AUC Score):")
        model_name = st.selectbox("Select Model", ["Logistic Regression", "KNN", "Random Forest", "Decision Tree", "XGBoost", "Neural Network", "CNN"])

        scaler = MinMaxScaler()
        input_data_scaled = scaler.fit_transform(test_data.drop(columns=['churn']))

        if model_name in ["Logistic Regression", "KNN", "Random Forest", "Decision Tree", "XGBoost"]:
            predictions = models[model_name].predict_proba(input_data_scaled)[:, 1]
        elif model_name in ["Neural Network", "CNN"]:
            if model_name == "CNN":
                input_data_scaled = input_data_scaled.reshape(input_data_scaled.shape[0], input_data_scaled.shape[1], 1)
            predictions = models[model_name].predict(input_data_scaled)

        actual_labels = test_data['churn']
        roc_auc = roc_auc_score(actual_labels, predictions)
        st.write(f"ROC AUC Score for {model_name}: {roc_auc}")

        fpr, tpr, _ = roc_curve(actual_labels, predictions)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(5.8, 4.1))
        ax.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        st.pyplot(fig)

    elif task == "Prediction History":
        st.subheader("Prediction History")
        prediction_history = load_prediction_history()
        st.dataframe(prediction_history)

if __name__ == "__main__":
    main()


import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
from keras.models import load_model
import matplotlib.pyplot as plt

# Load các mô hình và dữ liệu test
logistic_regression_model = pickle.load(open("logistic_regression_model.pkl", "rb"))
knn_model = pickle.load(open("knn_model.pkl", "rb"))
random_forest_model = pickle.load(open("random_forest_model.pkl", "rb"))
decision_tree_model = pickle.load(open("decision_tree_model.pkl", "rb"))
xgboost_model = pickle.load(open("xgboost_model.pkl", "rb"))
nn_model = load_model("neural_network_model.keras")
cnn_model = load_model("cnn_model.keras")
test_data = pd.read_csv("test.csv")

# Hàm để dự đoán churn bằng các mô hình
def predict_churn(input_data, model_name):
    # Tạo scaler dựa trên tập dữ liệu test
    scaler = MinMaxScaler()
    scaler.fit(test_data.drop(columns=['churn']))
    # Scaling dữ liệu
    input_data_scaled = scaler.transform(input_data)

    # Dự đoán churn với model tương ứng
    if model_name == "Logistic Regression":
        prediction = logistic_regression_model.predict_proba(input_data_scaled)[:, 1]
    elif model_name == "KNN":
        prediction = knn_model.predict_proba(input_data_scaled)[:, 1]
    elif model_name == "Random Forest":
        prediction = random_forest_model.predict_proba(input_data_scaled)[:, 1]
    elif model_name == "Decision Tree":
        prediction = decision_tree_model.predict_proba(input_data_scaled)[:, 1]
    elif model_name == "XGBoost":
        prediction = xgboost_model.predict_proba(input_data_scaled)[:, 1]
    elif model_name == "Neural Network":
        prediction = nn_model.predict(input_data_scaled)
    elif model_name == "CNN":
        input_data_scaled = input_data_scaled.reshape(input_data_scaled.shape[0], input_data_scaled.shape[1], 1)
        prediction = cnn_model.predict(input_data_scaled)

    return prediction

def main():
    # Tạo giao diện người dùng với Streamlit
    st.title('Telecom Customer Churn Prediction WEBAPP')
    st.write('This webapp will help you predict customer churn based on a Telecom Customer Churn Prediction.'
            ' You will predict the churn rate of customers in a telecom company using a stored model '
            ' based on Logistic Regression, KNN, Random Forest, Decision Tree, XGBoost, Neural Network, CNN.'
            ' To check the accuracy of the classifier, click on the Performance on Test Dataset button in the sidebar.'
            ' To predict, select the model you want to use from the dropdown box in the sidebar after choosing the user input data.')


    st.sidebar.title('User Input')

    task = st.sidebar.radio("Select Task", ["Predict", "Performance on Test Dataset"])

    if task == "Predict":
        # Nhập dữ liệu từ người dùng
        tenure = st.sidebar.number_input("Tenure", value=test_data['tenure'].mean())
        PhoneService = st.sidebar.selectbox("Phone Service", [" ", "Yes", "No", "No Phone Service"])
        Contract = st.sidebar.selectbox("Contract", [" ", "Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.sidebar.selectbox('Paperless Billing', ['', 'Yes', 'No'])
        PaymentMethod = st.sidebar.selectbox('Payment Method',
                                             ['', 'Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                                              'Credit card (automatic)'])
        MonthlyCharges = st.sidebar.number_input('Monthly Charges', value=test_data['monthly_charges'].mean())
        TotalCharges = st.sidebar.number_input('Total Charges', value=test_data['total_charges'].mean())
        gender = st.sidebar.selectbox("Gender", ['', "Male", "Female"])
        SeniorCitizen = st.sidebar.selectbox('Senior Citizen', ['', 'Yes', 'No'])
        Partner = st.sidebar.selectbox('Partner', ['', 'Yes', 'No'])
        Dependents = st.sidebar.selectbox('Dependents', ['', 'Yes', 'No'])
        MultipleLines = st.sidebar.selectbox('Multiple Lines', ['', 'Yes', 'No', 'No phone service'])
        InternetService = st.sidebar.selectbox('Internet Service', ['', 'DSL', 'Fiber optic', 'No'])
        OnlineSecurity = st.sidebar.selectbox('Online Security', ['', 'Yes', 'No', 'No internet service'])
        OnlineBackup = st.sidebar.selectbox('Online Backup', ['', 'Yes', 'No', 'No internet service'])
        DeviceProtection = st.sidebar.selectbox('Device Protection', ['', 'Yes', 'No', 'No internet service'])
        TechSupport = st.sidebar.selectbox('Tech Support', ['', 'Yes', 'No', 'No internet service'])
        StreamingTV = st.sidebar.selectbox('Streaming TV', ['', 'Yes', 'No', 'No internet service'])
        StreamingMovies = st.sidebar.selectbox('Streaming Movies', ['', 'Yes', 'No', 'No internet service'])

        # Tạo DataFrame từ dữ liệu đầu vào
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
            "multiple_lines_No phone service": [1 if MultipleLines == "No Phone Service" else 0],
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

        # Kiểm tra xem dữ liệu đầu vào có đủ để dự đoán hay không
        if (PhoneService == " " or Contract == " " or PaperlessBilling == '' or PaymentMethod == '' or gender == ''
                or SeniorCitizen == '' or Partner == '' or Dependents == '' or MultipleLines == '' or
                InternetService == '' or OnlineSecurity == '' or OnlineBackup == '' or DeviceProtection == '' or
                TechSupport == '' or StreamingTV == '' or StreamingMovies == ''):
            st.warning("Please fill in all input fields to make a prediction.")
        else:
            # Lựa chọn model
            model_name = st.sidebar.selectbox("Select Model",
                                              ["Logistic Regression", "KNN", "Random Forest", "Decision Tree", "XGBoost",
                                               "Neural Network", "CNN"])

            # Code dự đoán
            if st.sidebar.button("Predict"):
                # Dự đoán
                prediction = predict_churn(input_data, model_name)
                # Hiển thị kết quả
                if isinstance(prediction, list) or isinstance(prediction, np.ndarray):
                    prediction = prediction[0] * 100  # Lấy xác suất dự đoán từ danh sách xác suất
                st.title('Prediction Result')
                st.write("Churn Probability:", round(float(prediction), 2), "%")  # Chuyển đổi thành số thực và làm tròn
                if prediction < 50:
                    st.success('NOT Churn')
                else:
                    st.error('CHURN')


    elif task == "Performance on Test Dataset":
        st.subheader("Performance on The Test Dataset (ROC AUC Score):")
        model_name = st.selectbox("Select Model",
                                  ["Logistic Regression", "KNN", "Random Forest", "Decision Tree", "XGBoost",
                                   "Neural Network", "CNN"])
        
        # Tạo scaler dựa trên tập dữ liệu test
        scaler = MinMaxScaler()
        scaler.fit(test_data.drop(columns=['churn']))
        input_data = test_data.drop(columns=['churn'])
        # Scaling dữ liệu
        input_data_scaled = scaler.transform(input_data)
    
        # Dự đoán churn với model tương ứng
        if model_name == "Logistic Regression":
            predictions = logistic_regression_model.predict_proba(input_data_scaled)[:, 1]
        elif model_name == "KNN":
            predictions = knn_model.predict_proba(input_data_scaled)[:, 1]
        elif model_name == "Random Forest":
            predictions = random_forest_model.predict_proba(input_data_scaled)[:, 1]
        elif model_name == "Decision Tree":
            predictions = decision_tree_model.predict_proba(input_data_scaled)[:, 1]
        elif model_name == "XGBoost":
            predictions = xgboost_model.predict_proba(input_data_scaled)[:, 1]
        elif model_name == "Neural Network":
            predictions = nn_model.predict(input_data_scaled)
        elif model_name == "CNN":
            input_data_scaled = input_data_scaled.reshape(input_data_scaled.shape[0], input_data_scaled.shape[1], 1)
            predictions = cnn_model.predict(input_data_scaled)
        
        actual_labels = test_data['churn']
        roc_auc = roc_auc_score(actual_labels, predictions)
        st.write(f"ROC AUC Score for {model_name}: {roc_auc}")
    
        # Vẽ ROC curve
        fpr, tpr, _ = roc_curve(actual_labels, predictions)
        roc_auc = auc(fpr, tpr)
    
        # Vẽ ROC curve
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


if __name__ == "__main__":
        main()

import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
from keras.models import load_model
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score
from PIL import Image


# Set page configuration
st.set_page_config(page_title='Customer Churn Prediction', page_icon=':bar_chart:', layout='wide')

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False


# Create a directory for saving prediction history
if not os.path.exists("prediction_history"):
    os.makedirs("prediction_history")


# Load models and test data
@st.cache_resource
def load_models():
    logistic_regression_model = pickle.load(open("logistic_regression_model.pkl", "rb"))
    knn_model = pickle.load(open("knn_model.pkl", "rb"))
    random_forest_model = pickle.load(open("random_forest_model.pkl", "rb"))
    decision_tree_model = pickle.load(open("decision_tree_model.pkl", "rb"))
    xgboost_model = pickle.load(open("xgboost_model.pkl", "rb"))
    nn_model = load_model("neural_network_model.keras")
    cnn_model = load_model("cnn_model.keras")
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
    test_data = pd.read_csv("test.csv")
    return test_data

models = load_models()
test_data = load_data()

# Function to predict churn using models
def predict_churn(input_data, model_name):
    if model_name in ["Logistic Regression", "KNN", "Random Forest", "Decision Tree", "XGBoost"]:
        prediction = models[model_name].predict_proba(input_data)[:, 1]
    elif model_name in ["Neural Network", "CNN"]:
        if model_name == "CNN":
            input_data = input_data.reshape(input_data.shape[0], input_data.shape[1], 1)
        prediction = models[model_name].predict(input_data)
    return prediction

def save_prediction(prediction_data):
    history_file = "history.csv"
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
        history_df = pd.concat([history_df, prediction_data], ignore_index=True)
    else:
        history_df = prediction_data
    history_df.to_csv(history_file, index=False)

def load_prediction_history():
    history_file = "history.csv"
    if os.path.exists(history_file):
        return pd.read_csv(history_file)
    else:
        return pd.DataFrame()

def one_hot_encode_and_scale(input_data):
    # One-hot encode the categorical features
    categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    input_data = pd.get_dummies(input_data, columns=[col for col in categorical_features if col in input_data.columns])

    # Ensure all expected columns are present
    expected_columns = test_data.drop(columns=['churn']).columns
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Ensure the input data has the same column order as the training data
    input_data = input_data[expected_columns]

    # Scale the data
    scaler = MinMaxScaler()
    scaler.fit(test_data.drop(columns=['churn']))
    input_data_scaled = scaler.transform(input_data)

    return input_data_scaled

def encode_manual_input(input_data):
    # Manually encode categorical features
    input_data['senior_citizen'] = input_data['SeniorCitizen'].apply(lambda x: 1 if x == 'Yes' else 0)
    input_data['phone_service_No'] = input_data['PhoneService'].apply(lambda x: 1 if x == 'No' else 0)
    input_data['phone_service_Yes'] = input_data['PhoneService'].apply(lambda x: 1 if x == 'Yes' else 0)
    input_data['phone_service_No Phone Service'] = input_data['PhoneService'].apply(lambda x: 1 if x == 'No Phone Service' else 0)
    input_data['contract_Month-to-month'] = input_data['Contract'].apply(lambda x: 1 if x == 'Month-to-month' else 0)
    input_data['contract_One year'] = input_data['Contract'].apply(lambda x: 1 if x == 'One year' else 0)
    input_data['contract_Two year'] = input_data['Contract'].apply(lambda x: 1 if x == 'Two year' else 0)
    input_data['paperless_billing_No'] = input_data['PaperlessBilling'].apply(lambda x: 1 if x == 'No' else 0)
    input_data['paperless_billing_Yes'] = input_data['PaperlessBilling'].apply(lambda x: 1 if x == 'Yes' else 0)
    input_data['gender_Female'] = input_data['gender'].apply(lambda x: 1 if x == 'Female' else 0)
    input_data['gender_Male'] = input_data['gender'].apply(lambda x: 1 if x == 'Male' else 0)
    input_data['partner_No'] = input_data['Partner'].apply(lambda x: 1 if x == 'No' else 0)
    input_data['partner_Yes'] = input_data['Partner'].apply(lambda x: 1 if x == 'Yes' else 0)
    input_data['dependents_No'] = input_data['Dependents'].apply(lambda x: 1 if x == 'No' else 0)
    input_data['dependents_Yes'] = input_data['Dependents'].apply(lambda x: 1 if x == 'Yes' else 0)
    input_data['multiple_lines_No'] = input_data['MultipleLines'].apply(lambda x: 1 if x == 'No' else 0)
    input_data['multiple_lines_No phone service'] = input_data['MultipleLines'].apply(lambda x: 1 if x == 'No phone service' else 0)
    input_data['multiple_lines_Yes'] = input_data['MultipleLines'].apply(lambda x: 1 if x == 'Yes' else 0)
    input_data['internet_service_DSL'] = input_data['InternetService'].apply(lambda x: 1 if x == 'DSL' else 0)
    input_data['internet_service_Fiber optic'] = input_data['InternetService'].apply(lambda x: 1 if x == 'Fiber optic' else 0)
    input_data['internet_service_No'] = input_data['InternetService'].apply(lambda x: 1 if x == 'No' else 0)
    input_data['online_security_No'] = input_data['OnlineSecurity'].apply(lambda x: 1 if x == 'No' else 0)
    input_data['online_security_No internet service'] = input_data['OnlineSecurity'].apply(lambda x: 1 if x == 'No internet service' else 0)
    input_data['online_security_Yes'] = input_data['OnlineSecurity'].apply(lambda x: 1 if x == 'Yes' else 0)
    input_data['online_backup_No'] = input_data['OnlineBackup'].apply(lambda x: 1 if x == 'No' else 0)
    input_data['online_backup_No internet service'] = input_data['OnlineBackup'].apply(lambda x: 1 if x == 'No internet service' else 0)
    input_data['online_backup_Yes'] = input_data['OnlineBackup'].apply(lambda x: 1 if x == 'Yes' else 0)
    input_data['device_protection_No'] = input_data['DeviceProtection'].apply(lambda x: 1 if x == 'No' else 0)
    input_data['device_protection_No internet service'] = input_data['DeviceProtection'].apply(lambda x: 1 if x == 'No internet service' else 0)
    input_data['device_protection_Yes'] = input_data['DeviceProtection'].apply(lambda x: 1 if x == 'Yes' else 0)
    input_data['tech_support_No'] = input_data['TechSupport'].apply(lambda x: 1 if x == 'No' else 0)
    input_data['tech_support_No internet service'] = input_data['TechSupport'].apply(lambda x: 1 if x == 'No internet service' else 0)
    input_data['tech_support_Yes'] = input_data['TechSupport'].apply(lambda x: 1 if x == 'Yes' else 0)
    input_data['streaming_tv_No'] = input_data['StreamingTV'].apply(lambda x: 1 if x == 'No' else 0)
    input_data['streaming_tv_No internet service'] = input_data['StreamingTV'].apply(lambda x: 1 if x == 'No internet service' else 0)
    input_data['streaming_tv_Yes'] = input_data['StreamingTV'].apply(lambda x: 1 if x == 'Yes' else 0)
    input_data['streaming_movies_No'] = input_data['StreamingMovies'].apply(lambda x: 1 if x == 'No' else 0)
    input_data['streaming_movies_No internet service'] = input_data['StreamingMovies'].apply(lambda x: 1 if x == 'No internet service' else 0)
    input_data['streaming_movies_Yes'] = input_data['StreamingMovies'].apply(lambda x: 1 if x == 'Yes' else 0)
    input_data['payment_method_Electronic check'] = input_data['PaymentMethod'].apply(lambda x: 1 if x == 'Electronic check' else 0)
    input_data['payment_method_Mailed check'] = input_data['PaymentMethod'].apply(lambda x: 1 if x == 'Mailed check' else 0)
    input_data['payment_method_Bank transfer (automatic)'] = input_data['PaymentMethod'].apply(lambda x: 1 if x == 'Bank transfer (automatic)' else 0)
    input_data['payment_method_Credit card (automatic)'] = input_data['PaymentMethod'].apply(lambda x: 1 if x == 'Credit card (automatic)' else 0)

    # Drop original categorical columns
    input_data.drop(columns=['SeniorCitizen', 'PhoneService', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                             'gender', 'Partner', 'Dependents', 'MultipleLines', 'InternetService',
                             'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                             'StreamingTV', 'StreamingMovies'], inplace=True)

    return input_data

def main():
    if not st.session_state.logged_in:
        # Create two columns
        col1, col2 = st.columns([3, 2])  # Adjust the ratio as needed

        # Column for login form
        with col1:
            st.markdown("<h2 style='color:darkblue;'>LOG IN</h2>", unsafe_allow_html=True)

            username = st.text_input("Username:")
            password = st.text_input("Password:", type="password")

            # Create the "Remember" checkbox and "Forgot Password?" link
            remember_col, forgot_password_col = st.columns([1, 2])
            with remember_col:
                remember = st.checkbox("Remember")
            with forgot_password_col:
                st.markdown("<a href='#'>Forgot Password?</a>", unsafe_allow_html=True)

            st.warning("username == Linh and password == Linh")

            # Log in button
            if st.button("Log in", key="login_button", help="Login with the provided credentials"):
                if username == "Linh" and password == "Linh":
                    st.session_state.logged_in = True
                    st.success("Login successful!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password.")

        # Column for image display
        with col2:
            image = Image.open("maxresdefault.jpg")
            st.image(image, width=600)


    else:
        st.title('Telecom Customer Churn Prediction WEBAPP')
        st.write('This webapp will help you predict customer churn based on a Telecom Customer Churn Prediction.'
                 ' You will predict the churn rate of customers in a telecom company using a stored model '
                 ' based on Logistic Regression, KNN, Random Forest, Decision Tree, XGBoost, Neural Network, CNN.'
                 ' To check the accuracy of the classifier, click on the Performance on Test Dataset button in the sidebar.'
                 ' To predict, select the model you want to use from the dropdown box in the sidebar after choosing the user input data.')

        st.sidebar.title('MENU')
        task = st.sidebar.radio("Select Task",
                                ["Predict", "File Upload Predict", "Performance on Test Dataset", "Prediction History"])

        if task == "Predict":
            st.subheader('Input Customer Details')
            col1, col2 = st.columns(2)

            with col1:
                tenure = st.number_input("Tenure", value=int(test_data['tenure'].mean()))
                PhoneService = st.selectbox("Phone Service", ["Yes", "No", "No Phone Service"])
                Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                PaperlessBilling = st.selectbox('Paperless Billing', ['Yes', 'No'])
                PaymentMethod = st.selectbox('Payment Method',
                                             ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                                              'Credit card (automatic)'])
                MonthlyCharges = st.number_input('Monthly Charges', value=float(test_data['monthly_charges'].mean()))
                use_calculated_total_charges = st.checkbox('Use calculated Total Charges')
                st.write("Total Charges = Monthly Charges * Tenure + Extra Cost (~100)")

                if use_calculated_total_charges:
                    extra_cost = 100
                    calculated_TotalCharges = MonthlyCharges * tenure + extra_cost
                    TotalCharges = st.number_input('Total Charges', value=float(calculated_TotalCharges))
                else:
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
                "gender": [gender],
                "SeniorCitizen": [SeniorCitizen],
                "Partner": [Partner],
                "Dependents": [Dependents],
                "PhoneService": [PhoneService],
                "MultipleLines": [MultipleLines],
                "InternetService": [InternetService],
                "OnlineSecurity": [OnlineSecurity],
                "OnlineBackup": [OnlineBackup],
                "DeviceProtection": [DeviceProtection],
                "TechSupport": [TechSupport],
                "StreamingTV": [StreamingTV],
                "StreamingMovies": [StreamingMovies],
                "Contract": [Contract],
                "PaperlessBilling": [PaperlessBilling],
                "PaymentMethod": [PaymentMethod]
            })

            input_data_encoded = encode_manual_input(input_data)

            model_name = st.sidebar.selectbox("Select Model",
                                              ["Logistic Regression", "KNN", "Random Forest", "Decision Tree", "XGBoost",
                                               "Neural Network", "CNN"])

            if st.sidebar.button("Predict"):
                input_data_scaled = one_hot_encode_and_scale(input_data_encoded)
                prediction = predict_churn(input_data_scaled, model_name)
                prediction = prediction[0] * 100  # Convert to percentage
                st.subheader('Prediction Result')
                st.write("Churn Probability:", round(float(prediction), 2), "%")
                if prediction < 50:
                    st.success('NOT Churn')
                else:
                    st.error('CHURN')

                # Save prediction result
                prediction_data = input_data.copy()
                prediction_data["churn_probability"] = round(float(prediction), 2)
                save_prediction(prediction_data)

        elif task == "File Upload Predict":
            st.subheader('Upload Customer Data')
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file:
                input_data = pd.read_csv(uploaded_file)
                input_data_encoded = encode_manual_input(input_data)
                input_data_scaled = one_hot_encode_and_scale(input_data_encoded)

                model_name = st.sidebar.selectbox("Select Model",
                                                  ["Logistic Regression", "KNN", "Random Forest", "Decision Tree",
                                                   "XGBoost", "Neural Network", "CNN"])

                if st.sidebar.button("Predict"):
                    if model_name == "CNN":
                        input_data_scaled = input_data_scaled.reshape(input_data_scaled.shape[0],
                                                                      input_data_scaled.shape[1], 1)
                    predictions = predict_churn(input_data_scaled, model_name)
                    input_data['churn_probability'] = predictions * 100  # Convert to percentage
                    st.write(input_data)

                    # Download prediction results
                    csv = input_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name='predictions.csv',
                        mime='text/csv',
                    )


        elif task == "Performance on Test Dataset":
            st.subheader("Performance on The Test Dataset (ROC AUC Score and F1-score):")

            model_name = st.selectbox("Select Model",
                                      ["Logistic Regression", "KNN", "Random Forest", "Decision Tree", "XGBoost",
                                       "Neural Network", "CNN"])

            input_data_scaled = one_hot_encode_and_scale(test_data.drop(columns=['churn']))

            if model_name == "CNN":
                input_data_scaled = input_data_scaled.reshape(input_data_scaled.shape[0], input_data_scaled.shape[1], 1)

            predictions = predict_churn(input_data_scaled, model_name)
            actual_labels = test_data['churn']
            roc_auc = roc_auc_score(actual_labels, predictions)
            st.write(f"ROC AUC Score for {model_name}: {roc_auc}")
            fpr, tpr, _ = roc_curve(actual_labels, predictions)
            roc_auc = auc(fpr, tpr)
            f1 = f1_score(actual_labels, predictions.round())
            st.write(f"F1-score for {model_name}: {f1}")
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

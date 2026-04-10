import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64 

def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
    return href

st.title("❤️HealthScope-Heart Disease Predictor")
tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model Information'])

with tab1:
    st.markdown("### Enter Patient Details")    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=150)
        sex = st.selectbox("Sex", ["Male", "Female"])
        # Fixed: Added missing comma between Atypical Angina and Non-Anginal Pain
        chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    with col2:    
        cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
        fasting_bs = st.selectbox("Fasting Blood Sugar", ["<=120 mg/dl", ">120 mg/dl"])
        resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    with col3:    
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
        oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
        st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    
    # Convert categorical input to numeric
    sex_num = 0 if sex == "Male" else 1
    # Order must match exactly how the model was trained
    cp_num = ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"].index(chest_pain)
    fbs_num = 1 if fasting_bs == ">120 mg/dl" else 0
    ecg_num = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    ex_angina_num = 1 if exercise_angina == "Yes" else 0
    slope_num = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # FIXED: Column names now match the "Feature names seen at fit time"
    input_data = pd.DataFrame({   
        'Age': [age],
        'Sex': [sex_num],
        'ChestPainType': [cp_num],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fbs_num],
        'RestingECG': [ecg_num],
        'MaxHR': [max_hr],
        'ExerciseAngina': [ex_angina_num],
        'Oldpeak': [oldpeak],
        'ST_Slope': [slope_num]
    })

    algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']
    modelnames = ['Dtree.pkl', 'LogisticRegression.pkl', 'RandomForest.pkl', 'SVM.pkl']

    def predict_heart_disease(data):
        preds = []
        for modelname in modelnames:
            with open(modelname, 'rb') as f:
                model = pickle.load(f)
                prediction = model.predict(data)
                preds.append(prediction)
        return preds
    
    if st.button("Submit"):
        st.subheader('Results....')
        
        results = predict_heart_disease(input_data)

        res_col1, res_col2 = st.columns(2)

        for i in range(len(results)):

            target_col = res_col1 if i % 2 == 0 else res_col2
            
            with target_col:
                st.markdown(f"#### {algonames[i]}")
                if results[i][0] == 0:
                    st.success("No heart disease detected.")
                else:
                    st.error("Heart disease detected.")
                st.write("")

with tab2:
    st.title("Upload CSV File")
    st.subheader("Instructions to note before uploading the file:")    
    st.info("""
        1. No NaN values allowed.
        2. Total 11 features in specific order.
        3. Feature names must be: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope.
    """)
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        input_data_bulk = pd.read_csv(uploaded_file)
        
        # Load one model for bulk prediction
        with open('LogisticRegression.pkl', 'rb') as f:
            model = pickle.load(f)

        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 
                            'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

        if all(col in input_data_bulk.columns for col in expected_columns):
            # Select only the features needed for prediction (ignoring extra columns)
            features = input_data_bulk[expected_columns]
            
            # Predict for the whole dataframe at once (much faster than a loop)
            input_data_bulk['Prediction LR'] = model.predict(features)

            st.subheader("Predictions:")
            st.write(input_data_bulk)
            st.markdown(get_binary_file_downloader_html(input_data_bulk), unsafe_allow_html=True)
        else:
            st.warning("Please make sure the uploaded CSV file has the exact correct column names.")     
    else:
        st.info("Upload a CSV file to get predictions.")

with tab3:
    import plotly.express as px
    data_acc = {'Decision Trees': 80.97, 'Logistic Regression': 85.86, 'Random Forest': 84.23, 'Support Vector Machine': 84.22}
    df_acc = pd.DataFrame(list(data_acc.items()), columns=['Models', 'Accuracies'])
    fig = px.bar(df_acc, y='Accuracies', x='Models', color='Models')
    st.plotly_chart(fig)
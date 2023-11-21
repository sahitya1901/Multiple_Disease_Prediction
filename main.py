# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:29:53 2023

@author: ACER
"""

import joblib
import streamlit as st
from streamlit_option_menu import option_menu

#loading the saved models
#local
#diabetes_model = joblib.load('D:\Multiple_Disease_Prediction\diabetes_model.joblib')
#diabetes_scaler = joblib.load('D:\Multiple_Disease_Prediction\diabetes_scaler.joblib')
#parkinsons_model = joblib.load('D:\Multiple_Disease_Prediction\parkinsons_model.joblib')
#parkinsons_scaler = joblib.load('D:\Multiple_Disease_Prediction\parkinsons_scaler.joblib')
#heart_model = joblib.load('D:\Multiple_Disease_Prediction\heart_model.joblib')

#For Deployment
diabetes_model = joblib.load('diabetes_model.joblib')
diabetes_scaler = joblib.load('diabetes_scaler.joblib')
parkinsons_model = joblib.load('parkinsons_model.joblib')
parkinsons_scaler = joblib.load('parkinsons_scaler.joblib')
heart_model = joblib.load('heart_model.joblib')

#sidebar for navigate
with st.sidebar:
    selected=option_menu('Multiple Disease Prediction System',['Diabetes Prediction','Heart Disease Prediction','Parkinsons Prediction'],icons=['activity','heart','person'],default_index=0)
    
#Diabetes Prediction Page
if selected=='Diabetes Prediction':
    
    #Page Title
    st.title('Diabetes Prediction using ML')
    
    #input
    #columns for input fields
    col1,col2,col3=st.columns(3)
    with col1:
        Pregnancies=st.number_input('Number of Pregancies',min_value=0,step=1)
    with col2:  
        Glucose=st.number_input('Glucose Level',min_value=0,step=1)
    with col3:
        BloodPressure=st.number_input('Blood Pressure Value',min_value=0,step=1)
    with col1:
        SkinThickness=st.number_input('Skin Thickness Value',min_value=0,step=1)
    with col2:
        Insulin=st.number_input('Insulin Level',min_value=0,step=1)
    with col3:
        BMI=st.number_input('BMI value',min_value=0.0,step=1e-1,format="%.1f")
    with col1:
        DiabetesPedigreeFunction=st.number_input('Diabetes Pedigree Function Value',min_value=0.000,step=1e-3,format="%.3f")
    with col2:
        Age=st.number_input('Age of the Person',min_value=1,step=1)
    
    #Prediction 
    diab_diagonsis=''
    if st.button('Diabetes Test Result'):
        input_data=[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
        std_data=diabetes_scaler.transform([input_data])
        diab_prediction=diabetes_model.predict(std_data)
        if diab_prediction[0]==0:
            diab_diagonsis='The person is not diabetic'
        else:
            diab_diagonsis='The person is diabetic'
    st.success(diab_diagonsis)
    
    
#Heart Disease Prediction Page
if selected=='Heart Disease Prediction':
    
    #Page Title
    st.title('Heart Disease Prediction using ML')
    
    #input
    #columns for input fields
    col1,col2,col3=st.columns(3)
    with col1:
        Age=st.number_input('Age',min_value=1,step=1)
    with col2:  
        Sex=st.number_input('Sex (0-female,1-male)',min_value=0,max_value=1,step=1)
    with col3:
        Cp=st.number_input('Chest Pain Types',min_value=0,step=1)
    with col1:
        Trestbps=st.number_input('Rest Blood Pressure',min_value=0,step=1)
    with col2:
        Chol=st.number_input('Serum Cholesterol in mg/dl',min_value=0,step=1)
    with col3:
        Fbs=st.number_input('Fasting Blood Sugar >120 mg/dl',min_value=0,step=1)
    with col1:
        Restecg=st.number_input('Resting Electrocardiographic Results',min_value=0,step=1)
    with col2:
        Thalach=st.number_input('Maximum Heart Rate Achieved',min_value=0,step=1)
    with col3:
        Exang=st.number_input('Exercise Induced Angina',min_value=0,step=1)
    with col1:
        Oldpeak=st.number_input('ST Depression induced by exercise',min_value=0.0,step=1e-1,format="%.1f")
    with col2:
        Slope=st.number_input('Slope of the peak exercise ST Segment',min_value=0,step=1)
    with col3:
        Ca=st.number_input('Major vessels colored by flouroscopy',min_value=0,step=1)
    with col1:
        Thal=st.number_input('Thal (0-normal,1-fixed defect,2-reversable defect)',min_value=0,step=1)
      
    #Prediction 
    heart_diagonsis=''
    if st.button('Heart Test Result'):
        input_data=[Age,Sex,Cp,Trestbps,Chol,Fbs,Restecg,Thalach,Exang,Oldpeak,Slope,Ca,Thal]
        heart_prediction=heart_model.predict([input_data])
        if heart_prediction[0]==0:
            heart_diagonsis='The person does not have a heart disease'
        else:
            heart_diagonsis='The person has a heart disease'
    st.success(heart_diagonsis)
    
    
#Parkinsons Prediction Page
if selected=='Parkinsons Prediction':
    
    #Page Title
    st.title('Parkinsons Prediction using ML')
    
    #input
    #columns for input fields
    col1,col2,col3=st.columns(3)
    with col1:
        MDVP_Fo=st.number_input('MDVP:Fo(Hz)',min_value=0.000,step=1e-3,format="%.3f")
    with col2:  
        MDVP_Fhi=st.number_input('MDVP:Fhi(Hz)',min_value=0.000,step=1e-3,format="%.3f")
    with col3:
        MDVP_Flo=st.number_input('MDVP:Flo(Hz)',min_value=0.000,step=1e-3,format="%.3f")
    with col1:
        MDVP_Jitter_perc=st.number_input('MDVP:Jitter(%)',min_value=0.00000,step=1e-5,format="%.5f")
    with col2:
        MDVP_Jitter_Abs=st.number_input('MDVP:Jitter(Abs)',min_value=0.00000,step=1e-5,format="%.5f")
    with col3:
        MDVP_RAP=st.number_input('MDVP:RAP',min_value=0.00000,step=1e-5,format="%.5f")
    with col1:
        MDVP_PPQ=st.number_input('MDVP:PPQ',min_value=0.00000,step=1e-5,format="%.5f")
    with col2:
        Jitter_DDP=st.number_input('Jitter:DDP',min_value=0.00000,step=1e-5,format="%.5f")
    with col3:
        MDVP_Shimmer=st.number_input('MDVP:Shimmer',min_value=0.00000,step=1e-5,format="%.5f")
    with col1:
        MDVP_Shimmer_dB=st.number_input('MDVP:Shimmer(dB)',min_value=0.000,step=1e-3,format="%.3f")
    with col2:
        Shimmer_APQ3=st.number_input('Shimmer:APQ3',min_value=0.00000,step=1e-5,format="%.5f")
    with col3:
        Shimmer_APQ5=st.number_input('Shimmer:APQ5',min_value=0.00000,step=1e-5,format="%.5f")
    with col1:
        MDVP_APQ=st.number_input('MDVP:APQ',min_value=0.00000,step=1e-5,format="%.5f")
    with col2:
        Shimmer_DDA=st.number_input('Shimmer:DDA',min_value=0.00000,step=1e-5,format="%.5f")
    with col3:
        NHR=st.number_input('NHR',min_value=0.00000,step=1e-5,format="%.5f")
    with col1:
        HNR=st.number_input('HNR',min_value=0.000,step=1e-3,format="%.3f")
    with col2:
        RPD=st.number_input('RPD',min_value=0.000000,step=1e-6,format="%.6f")
    with col3:
        DFA=st.number_input('DFA',min_value=0.000000,step=1e-6,format="%.6f")
    with col1:
        spread1=st.number_input('spread1',step=1e-6,format="%.6f")
    with col2:
        spread2=st.number_input('spread2',min_value=0.00000,step=1e-6,format="%.6f")
    with col3:
        D2=st.number_input('D2',min_value=0.000000,step=1e-6,format="%.6f")
    with col1:
        PPE=st.number_input('PPE',min_value=0.000000,step=1e-6,format="%.6f")

    #Prediction 
    parkinsons_diagonsis=''
    if st.button('Parkinsons Test Result'):
        input_data=[MDVP_Fo,MDVP_Fhi,MDVP_Flo,MDVP_Jitter_perc,MDVP_Jitter_Abs,MDVP_RAP,MDVP_PPQ,Jitter_DDP,MDVP_Shimmer,MDVP_Shimmer_dB,Shimmer_APQ3,Shimmer_APQ5,MDVP_APQ,Shimmer_DDA,NHR,HNR,RPD,DFA,spread1,spread2,D2,PPE]
        std_data=parkinsons_scaler.transform([input_data])
        parkinsons_prediction=parkinsons_model.predict(std_data)
        if parkinsons_prediction[0]==0:
            parkinsons_diagonsis='The person does not have parkinsons disease'
        else:
            parkinsons_diagonsis='The person has parkinsons disease'
    st.success(parkinsons_diagonsis)
#from typing import no_type_check
import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
from statistics import mode



def main_prediction():
    '''
    # main prediction function
    # accepts input from user & predicts if the student gets placed or not
    '''
    # title of the page
    st.title("Employee Attrition Prediction - ML API")
        
    # user inputs
    Age = st.number_input("Age",19,60)
    JobLevel = st.number_input("JobLevel",1,5)
    YearsAtCompany = st.number_input("YearsAtCompany",0,50)
   
    # user input preprocesing for ML model input
    user_ip = [[Age,JobLevel,YearsAtCompany]]
   
    # pickle files of the models
    pickled_model = open("attrition.pkl", "rb")
    classifier = pickle.load(pickled_model)
    result = classifier.predict(user_ip)[0]

    if result == 1:
        prediction = 'Attrition'
    else:
        prediction = 'No attrition'

    # button print result
    if st.button("prediction"):
        st.success(f"Employee possibly {prediction}")        

if __name__ == "__main__":
    main_prediction()

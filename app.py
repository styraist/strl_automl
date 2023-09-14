import streamlit as st
import pandas as pd
import os
import plotly.express as px

# profiling
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

# ML
from pycaret.classification import *
from pycaret.regression import *
# from pycaret.clustering import *


if os.path.exists("./dataset.csv"):
    df = pd.read_csv("dataset.csv", index_col=None)
else:
    df = pd.DataFrame() # bir dataframe sağlanmamışsa varsayılan dataframe


with st.sidebar:
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download"])
    st.info("Bu uygulama Streamlit, Pandas Profiling ve PyCaret kullanarak bir AutoML pipeline oluşturmanıza olanak tanır.")
   

if choice == "Upload":
    st.title("Modelleme için verilerinizi yükleyin!")
    file = st.file_uploader("Veri setinizi buraya yükleyin.")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("dataset.csv", index=None)
        st.dataframe(df)
        
        
if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)
    
    
if choice == "Modelling":
    chosen_target = st.selectbox("Select Your Target", df.columns)
    if chosen_target and st.button("Run Modelling"):
        setup(df, target=chosen_target, silent=True)
        setup_df = pull()

        best_model = compare_models()
        compare_df = pull()
        st.info("ML Model")
        save_model(best_model, 'best_model')
        
        st.dataframe(compare_df)
       
        
if choice == "Download":
    if os.path.exists('best_model.pkl'):
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name = "best_model.pkl")
    else:
        st.warning("No model has been saved yet. Please run modelling first.")
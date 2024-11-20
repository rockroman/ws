import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_ames_data, load_pkl_file


def ml_price_prediction_page():
    """
    Displays ML pipeline, feature importance and ML and regression
    performance plots 
    """
    # load price pipeline files
    version = 'v4'
    price_pipe = load_pkl_file(f"outputs/ml_pipeline/predict_price/{version}/regression_pipeline.pkl")
    price_feat_importance = plt.imread(f"outputs/ml_pipeline/predict_price/{version}/features_importance.png")
    X_train = pd.read_csv(f"outputs/ml_pipeline/predict_price/{version}/X_train.csv")
    X_test = pd.read_csv(f"outputs/ml_pipeline/predict_price/{version}/X_test.csv")
    y_train =  pd.read_csv(f"outputs/ml_pipeline/predict_price/{version}/y_train.csv")
    y_test =  pd.read_csv(f"outputs/ml_pipeline/predict_price/{version}/y_test.csv")

 

    st.write("### ML Pipeline: Predict House Price")    
    # display pipeline training summary conclusions
    st.info(
        f"* To answer BR2, a Regressor model was trained with a pipeline. The goald was\
             an accuracy of at least 0.75 in predicting the sales price of a property that has a set of\
                 attributes. \n"
        f"* The pipeline performance for the best model on the train and test set is\
             R2 == 0.XXXXXXXXX and R2 == 0.XXXXXXXXXXXX respectively.\n"
        f"* The pipeline steps, best features determined, feature importance plot, pipeline performance and regression performance are shown below. "
       )
    st.write("---")
    
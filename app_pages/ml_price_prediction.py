import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from src.data_management import load_ames_data, load_pkl_file
from src.machine_learning.evaluate_reg import model_performance, pipeline_results


def load_pkl_file(file_path):
    """
    Loads a pickled file (model) using joblib from the given file path.
    """
    try:
        model = joblib.load(file_path)
        return model
    except Exception as e:
        print(f"Error loading model from {file_path}: {e}")
        return None


def ml_price_prediction_page():
    """
    Displays ML pipeline, feature importance and ML and regression
    performance plots 
    """
    # load price pipeline files
    version = 'v4'
    price_pipe = load_pkl_file(f"outputs/ml_pipeline/predict_price/v4/regression_pipeline.pkl")
    price_feat_importance = plt.imread(f"outputs/ml_pipeline/predict_price/v4/features_importance.png")
    X_train = pd.read_csv(f"outputs/ml_pipeline/predict_price/v4/X_train.csv")
    X_test = pd.read_csv(f"outputs/ml_pipeline/predict_price/v4/X_test.csv")
    y_train =  pd.read_csv(f"outputs/ml_pipeline/predict_price/v4/y_train.csv")
    y_test =  pd.read_csv(f"outputs/ml_pipeline/predict_price/v4/y_test.csv")

 
 
    st.write("### ML Pipeline: Predict House Price")    
        
    st.write(
    f"* To answer BR2, a Regressor model was trained with a pipeline. The goal was "
    f"an accuracy of at least 0.75 in predicting the sales price of a property that has a set of "
    f"attributes. The R2 for the train set and test set are as follows:"
    )

    st.success(f"Train Set R2 = 0.84")
    st.success(f"Test Set R2 = 0.77")

    st.write(f"* Both the Train Set R2 of 0.84 and the Test Set R2 of 0.77 exceed the required threshold of 0.75, indicating a successful model performance on both sets")

    st.write(
    f"* The pipeline steps, best features determined, feature importance plot, pipeline performance and "
    f"regression performance are shown below."
    )    
    
    st.write("---")
    
    # Add the pipeline steps
    st.write("* ML pipeline to predict sales prices of houses ")
    st.code(price_pipe)
    st.write("---")

    # display the five best features
    st.write("* The features the model was trained and their importance")
    st.write(X_train.columns.to_list())
    st.image(price_feat_importance)
    st.write("---")

    # Evaluate performance on both sets
    st.write("### Pipeline Performance")
    model_performance(X_train=X_train, y_train=y_train,
                      X_test=X_test, y_test=y_test,
                      pipeline=price_pipe)
    

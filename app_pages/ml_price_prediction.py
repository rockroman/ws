import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from src.data_management import load_ames_data, load_pkl_file
from src.machine_learning.evaluate_reg import regression_performance, regression_evaluation, regression_evaluation_plots

st.write(f"Does the directory exist? {os.path.isdir('outputs/ml_pipeline/predict_price/v4')}")


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


    # Displaying the pipeline performance
    st.write("### Pipeline Performance")
    st.write("##### Performance goal of the predictions:\n")
    st.write("* We agreed with the client an R2 score of at least 0.75 on the train set as well as on the test set.")
    st.write(f"* Our ML pipeline performance shows that our model performance metrics have been successfully satisfied.")

    # Perform performance evaluation
    regression_performance(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pipeline=price_pipe)

    # Display the regression performance plots
    st.write("### Regression Performance Plots")
    st.write("* The regression performance plots below indicate that our model, in most part, is able to predict sale prices well. The model looks less effective for houses with high prices though.")

    # Show the regression evaluation plots
    regression_evaluation_plots(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pipeline=price_pipe)

import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from src.data_management import load_ames_data, load_pkl_file
from src.machine_learning.evaluate_reg import regression_performance


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
        f"* To answer the BR2, a Regressor model was trained and the pipeline tuned aiming \
             for at least 0.75 accuracy in predicting the sales price of a property with a set of\
                 attributes. \n"
        f"* The pipeline performance for the best model on the train and test set is\
             R2 = 0.88 and R2 = 0.79 respectively.\n"
        f"* We present the pipeline steps, best features list along with feature importance plot, pipeline performance and regression performance report below. "
       )
    st.write("---")

    # show pipeline steps
    st.write("* ML pipeline to predict sales prices of houses ")
    st.code(price_pipe)
    st.write("---")

    # show best features
    st.write("* The features the model was trained and their importance")
    st.write(X_train.columns.to_list())
    st.image(price_feat_importance)
    st.write("---")


    st.write("### Pipeline Performance")
    st.write("##### Performance goal of the predictions:\n")
    st.write("* We agreed with the client an R2 score of at least 0.75 on the train set as well as on the test set.")
    st.write(f"* Our ML pipeline performance shows that our model performance\
         metrics have been successfully satisfied.")
    regression_performance(X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test,
                        pipeline=price_pipe) 

    # Display the regression performance plot
    st.write("### Regression Performance Plots")
    st.write(
    "* The regression performance plots below indicate that the model with the best features, "
    "can predict sale prices well. For houses with higher prices, the model does however"
    "look to be less dependable"
    )

    # Path to the image
    image_path = 'docs/plots/regression_performance.png'
    st.image(image_path, caption="Regression Performance", use_column_width=True)

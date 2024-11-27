import streamlit as st
import pandas as pd
import joblib
from src.data_management import load_inherited_houses_data  # Ensure this imports your existing function
from src.machine_learning.predictive_analysis_functions import predict_price  # Your predict function

def load_model():
    """Loads the regression pipeline."""
    file_path = "outputs/ml_pipeline/predict_price/v4"
    return joblib.load(f"{file_path}/regression_pipeline.pkl")

def sales_price_prediction_page():
    """
    Streamlit page to predict sales price of inherited houses.
    """
    # Load the pipeline defined
    regression_pipe = load_model()

    # Load the dataset of inherited houses
    X_inherited = load_inherited_houses_data()

    # the 5 most important features defined in ML model are:
    house_features = ['OverallQual', 'GrLivArea', 'GarageArea', 'YearBuilt', 'TotalBsmtSF']

    # I want to filter the inherited houses data on ONLY the important features determined
    X_inherited_filtered = X_inherited[house_features]

    # Display the filtered data (optional, if you want to show only these features to the user)
    st.write("#### Inherited Houses | Most important features")
    st.write(X_inherited_filtered)

    # Prediction of house prices with the pre-trained model using the 5 most important features
    predicted_prices = predict_price(X_inherited_filtered, house_features, regression_pipe)

    # Add the predicted prices to the dataframe
    X_inherited['PredictedSalePrice'] = predicted_prices

    # Display the filtered features with predicted prices for each house
    st.write("#### Predicted Sale Prices for Inherited Houses")
    st.dataframe(X_inherited[['OverallQual', 'GrLivArea', 'GarageArea', 'YearBuilt', 'TotalBsmtSF', 'PredictedSalePrice']])

    # The sum of predicted house prices calculated as follows
    total_price = round(X_inherited['PredictedSalePrice'].sum())

    # Display the total predicted sale price
    st.write(f"### Total Predicted Sale Price for All Inherited Houses: **${total_price}**")

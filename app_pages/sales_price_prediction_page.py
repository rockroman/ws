import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from src.data_management import load_inherited_houses_data, load_pkl_file, load_ames_data
from src.machine_learning.predictive_analysis_functions import predict_inherited_house_price, predict_price
import joblib

# Function to load the pre-trained model (regression pipeline)
def load_model():
    """
    Loads the trained model from the saved file.
    """
    file_path = "outputs/ml_pipeline/predict_price/v4/regression_pipeline.pkl"
    
    # Load the saved model
    model = joblib.load(file_path)
    
    return model


def sales_price_prediction_page():
    """
    The main function to display the Streamlit page for sales price prediction.
    """
    # Load the pre-trained regression pipeline model
    regression_pipe = load_model()

    # Your house feature column names (important features)
    house_features = ['OverallQual', 'GrLivArea', 'GarageArea', 'YearBuilt', 'TotalBsmtSF']

    st.write("### Predicting sales price of inherited houses (BR2)")

    # Load the inherited houses data
    X_inherited = load_inherited_houses_data()

    # Reduce inherited houses data down to only the important features listed above
    X_inherited = X_inherited[house_features]

    # Predict the sale price for each house
    summed_price = 0
    predicted_sale_price = []

    # Iterate through each house in the dataset
    for i in range(X_inherited.shape[0]):
        # Ensure the row has only the important features before prediction
        X_row = X_inherited.iloc[[i, ]][house_features]
        
        # Predict sale price using the loaded model
        pprice = predict_inherited_house_price(X_row, house_features, regression_pipe)
        
        # Round and collect the predictions
        predicted_sale_price.append(round(pprice))
        summed_price += pprice

    summed_price = round(summed_price)

    # Add the predicted sale prices to the dataframe
    X_inherited['PredictedSalePrice'] = predicted_sale_price

    # Display the predictions in Streamlit
    st.write(X_inherited.head())
    st.write(f"* Summed price: **${summed_price}** \n")

    # Prediction for live data input by the user
    st.subheader("Enter the house details to predict the potential price:")
    X_live = DrawInputsWidgets(house_features)

    if st.button('Predict House Sale Price'):
        live_price_prediction = predict_price(X_live, house_features, regression_pipe)
        st.write(f"Predicted Sale Price for Live Data: ${round(live_price_prediction)}")

    st.write("---")


def DrawInputsWidgets(house_features):
    """
    Create interactive input widgets for the user to enter live data for prediction.
    """
    # Load the Ames dataset for reference
    df = load_ames_data()
    percentageMin, percentageMax = 0.4, 2.0

    # Create an empty DataFrame for live data input
    X_live = pd.DataFrame([], index=[0])

    # Create the columns for the input widgets
    col1, col2, col3, col4, col5 = st.columns(5)

    # Ensure the relevant features are in the correct order
    relevant_features = ['OverallQual', 'GrLivArea', 'GarageArea', 'YearBuilt', 'TotalBsmtSF']

    # Create input widgets for each feature
    with col1:
        feature = relevant_features[0]
        st_widget = st.number_input(
            label=feature,
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )
        X_live[feature] = st_widget

    with col2:
        feature = relevant_features[1]
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min() * percentageMin),
            max_value=int(df[feature].max() * percentageMax),
            value=int(df[feature].median()),
            step=50
        )
        X_live[feature] = st_widget

    with col3:
        feature = relevant_features[2]
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min()),
            max_value=int(df[feature].max()),
            value=int(df[feature].median()),
            step=50
        )
        X_live[feature] = st_widget

    with col4:
        feature = relevant_features[3]
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min() * percentageMin),
            max_value=date.today().year,
            value=int(df[feature].median()),
            step=1
        )
        X_live[feature] = st_widget

    with col5:
        feature = relevant_features[4]
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min() * percentageMin),
            max_value=int(df[feature].max() * percentageMax),
            value=int(df[feature].median()),
            step=50
        )
        X_live[feature] = st_widget

    return X_live

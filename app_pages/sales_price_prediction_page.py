import streamlit as st
import pandas as pd
from datetime import date
import joblib
from src.data_management import load_ames_data
from src.data_management import load_inherited_houses_data
from src.machine_learning.predictive_analysis_functions import predict_price

def load_model():
    """Loads the regression pipeline."""
    file_path = "outputs/ml_pipeline/predict_price/v4"
    return joblib.load(f"{file_path}/regression_pipeline.pkl")

def sales_price_prediction_page():
    """
    Streamlit page to predict sales price of inherited houses and random houses based on user input.
    """
    # Load the pipeline
    regression_pipe = load_model()

    # Load inherited houses data for predictions
    X_inherited = load_inherited_houses_data()

    # Define the 5 important features
    house_features = ['OverallQual', 'GrLivArea', 'GarageArea', 'YearBuilt', 'TotalBsmtSF']

    # Filter the data to only include the important features
    X_inherited_filtered = X_inherited[house_features]

    # Display the filtered data (optional, if you want to show only these features to the user)
    st.write("#### Inherited Houses (Filtered Data for Prediction)")
    st.write(X_inherited_filtered)

    # Predict house prices using the loaded model and filtered features
    predicted_prices = predict_price(X_inherited_filtered, house_features, regression_pipe)

    # Add the predicted prices to the dataframe
    X_inherited['PredictedSalePrice'] = predicted_prices

    # Display the filtered features with predicted prices for each house
    st.write("#### Predicted Sale Prices for Inherited Houses")
    st.dataframe(X_inherited[['OverallQual', 'GrLivArea', 'GarageArea', 'YearBuilt', 'TotalBsmtSF', 'PredictedSalePrice']])

    # Calculate the sum of predicted prices
    total_price = round(X_inherited['PredictedSalePrice'].sum())

    # Display the total predicted sale price
    st.write(f"### Total Predicted Sale Price for All Inherited Houses: **${total_price}**")

    # Title for the random house prediction section
    st.write("#### Predict Sales Price for Your Own House")

    # Get user input for a custom house prediction
    X_live = DrawInputsWidgets(house_features)

    # Predict the price for the custom house
    price_prediction_live = predict_price(X_live, house_features, regression_pipe)

    # Display the prediction for the custom house
    st.write(f"### Predicted Sale Price for Your House: **${round(price_prediction_live[0])}**")


def DrawInputsWidgets(house_features):
    """
    Create interactive input widgets for the user.
    """
    # Load the Ames dataset for reference
    df = load_ames_data()
    percentageMin, percentageMax = 0.4, 2.0

    # Load the trained regression pipeline from the pickle file
    price_pipeline = joblib.load("outputs/ml_pipeline/predict_price/v4/regression_pipeline.pkl")

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
    
    if st.button('Calculate House Price'):
        # The predict price function is used to predict the price of the individual house
        predicted_price = predict_price(X_live, house_features, price_pipeline)
        
        # Display the predicted price
        st.write(f"**Predicted House Price:** ${round(predicted_price[0], 2):,}")
    
    return X_live

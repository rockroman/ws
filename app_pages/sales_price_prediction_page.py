import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from src.data_management import load_inherited_houses_data, load_pkl_file, load_ames_data
from src.machine_learning.predictive_analysis_functions import predict_inherited_house_price, predict_price


def sales_price_prediction_page():
    # Load the files for prediction
    version = 'v4'
    regression_pipe = load_pkl_file(f"outputs/ml_pipeline/predict_price/{version}/regression_pipeline.pkl")
    house_features = (
        pd.read_csv(f"outputs/ml_pipeline/predict_price/{version}/X_train.csv")
        .columns
        .to_list()
    )

    # I determined the following 5 most important features
    important_features = ['OverallQual', 'GrLivArea', 'GarageArea', 'YearBuilt', 'TotalBsmtSF']

    st.write("### Predicting sales price of inherited houses (BR2)")
    st.info(
        f"* **BR2** - The client is interested in predicting the house sale prices from her four inherited houses,"
        f" and any other house in Ames, Iowa."
    )

    # Load the inherited houses data
    X_inherited = load_inherited_houses_data()

    # Reduce inherited houses data down to only the important features listed above
    X_inherited = X_inherited[important_features]

    # Predict the sale price for each house
    summed_price = 0
    predicted_sale_price = []

    for i in range(X_inherited.shape[0]):
        # Prediction of sale price for each house
        pprice = predict_inherited_house_price(X_inherited.iloc[[i, ]], important_features, regression_pipe)
        predicted_sale_price.append(round(pprice))
        summed_price += pprice

    summed_price = round(summed_price)

    # Predicted sale price added to dataframe where 5 most important features already sit
    X_inherited['PredictedSalePrice'] = predicted_sale_price

    # Display the predictions
    st.write(X_inherited.head())
    st.write(
        f"* Summed price: **${summed_price}** \n"
        f"* Features used: **{X_inherited.columns.to_list()}**.\n"
        f" EXPLANATION / EXPLANATION "
        f" EXPLANATION / EXPLANATION"
    )

    # Widget to predict price for any house in the area based on impportant features
    st.subheader("Enter the house details to predict the potential price:")

    # Generate live data
    X_live = DrawInputsWidgets(house_features)

    live_price_prediction = predict_price(
        X_live, house_features, regression_pipe)

    # Prediction done on live data entered by the user
    if st.button('Predict House Sale Price'):
        price_prediction = predict_price(
            X_live, house_features, regression_pipe
        )

    st.write("---")
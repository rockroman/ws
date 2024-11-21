import streamlit as st
import pandas as pd
from datetime import date
from src.data_management import load_inherited_houses_data, load_pkl_file, load_ames_data
from src.machine_learning.predictive_analysis_functions import predict_inherited_house_price, predict_price

def sales_price_prediction_page():
    # Load the files for prediction
    version = 'v4'
    regression_pipe = load_pkl_file(f"outputs/ml_pipeline/predict_price/{version}/regression_pipeline.pkl")
    house_features = (pd.read_csv(f"outputs/ml_pipeline/predict_price/{version}/X_train.csv")
                     .columns
                     .to_list())

    # The 5 most important features determined are:
    important_features = ['OverallQual', 'GrLivArea', 'GarageArea', 'YearBuilt', 'TotalBsmtSF']

    st.write("### Predicting sales price of inherited houses (BR2)")
    st.info(
        f"* **BR2** - The client is interested in predicting the house sale prices from her four inherited houses,"
        f" and any other house in Ames, Iowa."
    )

    # Load the inherited houses data
    X_inherited = load_inherited_houses_data()

    # Reduce inherited houses data down to only the important features
    X_inherited = X_inherited[important_features]

    # Predict the sale price for each house
    summed_price = 0
    predicted_sale_price = []

    for i in range(X_inherited.shape[0]):
        # Prediction of sale price for each house
        pprice = predict_inherited_house_price(X_inherited.iloc[[i,]], important_features, regression_pipe)
        predicted_sale_price.append(round(pprice))
        summed_price += pprice

    summed_price = round(summed_price)

    # Predicted sale price added to dataframe where 5 most important features already sit
    X_inherited['PredictedSalePrice'] = predicted_sale_price

    # Display the predictions
    st.write(X_inherited.head())
    st.write(f"* Summed price: **${summed_price}** \n"
             f"* Features used: **{X_inherited.columns.to_list()}**.\n"
             f" EXPLANATION / EXPLANATION "
             f" EXPLANATION / EXPLANATION"
    )


def load_ames_data():
    df = pd.read_csv("outputs/datasets/collection/house_prices.csv")
    st.write("Columns in the dataset:", df.columns)
    return df


def create_input_widgets():
    
    df1 = load_ames_data()
    st.write(df1.head())
    if df1.empty:
        st.error("The dataset is empty!")
    return pd.DataFrame()

    percentageMin, percentageMax = 0.4, 2.0
    col1, col2, col3, col4, col5 = st.columns(5)

    # Create an empty DataFrame to hold input data
    X_live = pd.DataFrame([], index=[0])

    # Create widgets for the 5 features
    with col1:
        feature = "OverallQual"
        st_widget = st.number_input(
            label=feature,
            min_value=1,
            max_value=10,
            value=int(df1[feature].median()), 
            step=1
        )
        X_live[feature] = st_widget

    with col2:
        feature = "GrLivArea"
        st_widget = st.number_input(
            label=feature,
            min_value=int(df1[feature].min() * percentageMin),
            max_value=int(df1[feature].max() * percentageMax),  
            value=int(df1[feature].median()),
            step=100
        )
        X_live[feature] = st_widget

    with col3:
        feature = "GarageArea"
        st_widget = st.number_input(
            label=feature,
            min_value=int(df1[feature].min() * percentageMin),
            max_value=int(df1[feature].max() * percentageMax),
            value=int(df1[feature].median()),
            step=50
        )
        X_live[feature] = st_widget

    with col4:
        feature = "YearBuilt"
        st_widget = st.number_input(
            label=feature,
            min_value=int(df1[feature].min() * percentageMin),
            max_value=2024, 
            value=int(df1[feature].median()),
            step=1
        )
        X_live[feature] = st_widget

    with col5:
        feature = "TotalBsmtSF"
        st_widget = st.number_input(
            label=feature,
            min_value=int(df1[feature].min() * percentageMin),
            max_value=int(df1[feature].max() * percentageMax),
            value=int(df1[feature].median()),
            step=50
        )
        X_live[feature] = st_widget
    

    return X_live

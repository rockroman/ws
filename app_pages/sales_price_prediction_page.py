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

@st.cache
def load_ames_data():
    try:
        df = pd.read_csv("outputs/datasets/collection/house_prices.csv")
        st.write("Columns in the dataset:", df.columns)
        st.write("First few rows of the dataset:", df.head())
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()
    
    
    
def DrawInputsWidgets():
    
    # df = load_ames_data()
    # if df.empty:
    #     st.error("Dataset is empty or could not be loaded.")
    #     return

    df = pd.read_csv("test_data.csv")
    st.write(df.head())

    
    df.fillna(df.median(), inplace=True)

    percentageMin, percentageMax = 0.4, 2.0

    # List of top features to show in widgets
    top_features = ["OverallQual", "GrLivArea", "TotalBsmtSF", "YearBuilt", "1stFlrSF"]
    
    # Create columns for the input widgets
    col1, col2, col3, col4, col5 = st.columns(5)

    # Create empty DataFrame for live data
    X_live = pd.DataFrame([], index=[0])

    # Draw the widgets for each feature
    with col1:
        feature = top_features[0]
        if feature in df.columns:
            st_widget = st.number_input(
                label=feature,
                min_value=df[feature].min() * percentageMin,
                max_value=df[feature].max() * percentageMax,
                value=df[feature].median()
            )
            X_live[feature] = st_widget
        else:
            st.error(f"Feature '{feature}' not found in the dataset.")

    with col2:
        feature = top_features[1]
        if feature in df.columns:
            st_widget = st.number_input(
                label=feature,
                min_value=df[feature].min() * percentageMin,
                max_value=df[feature].max() * percentageMax,
                value=df[feature].median()
            )
            X_live[feature] = st_widget
        else:
            st.error(f"Feature '{feature}' not found in the dataset.")

    with col3:
        feature = top_features[2]
        if feature in df.columns:
            st_widget = st.number_input(
                label=feature,
                min_value=df[feature].min() * percentageMin,
                max_value=df[feature].max() * percentageMax,
                value=df[feature].median()
            )
            X_live[feature] = st_widget
        else:
            st.error(f"Feature '{feature}' not found in the dataset.")

    with col4:
        feature = top_features[3]
        if feature in df.columns:
            st_widget = st.number_input(
                label=feature,
                min_value=df[feature].min() * percentageMin,
                max_value=df[feature].max() * percentageMax,
                value=df[feature].median()
            )
            X_live[feature] = st_widget
        else:
            st.error(f"Feature '{feature}' not found in the dataset.")

    with col5:
        feature = top_features[4]
        if feature in df.columns:
            st_widget = st.number_input(
                label=feature,
                min_value=df[feature].min() * percentageMin,
                max_value=df[feature].max() * percentageMax,
                value=df[feature].median()
            )
            X_live[feature] = st_widget
        else:
            st.error(f"Feature '{feature}' not found in the dataset.")

    return X_live


if __name__ == "__main__":
    X_live = DrawInputsWidgets()
    if not X_live.empty:
        st.write("Live Data from Widgets:")
        st.write(X_live)

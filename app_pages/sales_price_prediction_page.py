import streamlit as st
import pandas as pd
from datetime import date
from src.data_management import load_inherited_houses_data, load_pkl_file
from src.machine_learning.predictive_analysis_functions import predict_inherited_house_price 

def sales_price_prediction_page():
	
	# load the files for prediction
	version = 'v4'
	regression_pipe = load_pkl_file(f"outputs/ml_pipeline/predict_price/{version}/regression_pipeline.pkl")
	house_features = (pd.read_csv(f"outputs/ml_pipeline/predict_price/{version}/X_train.csv")
					.columns
					.to_list()
					)


	st.write("### Predicting sales price of inherited houses (BR2)")
	st.info(
       f"* **BR2** - The client is interested in predicting the house sale prices from her four inherited houses,"
       f" and any other house in Ames, Iowa."
	)

	
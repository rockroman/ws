import streamlit as st
from app_pages.multipage import MultiPage

st.set_page_config(page_title="Heritage Housing Sale Price Predictor", page_icon="🖥️")

# the scripts for the five pages are loaded here
from app_pages.summary_page import summary_page_body
from app_pages.correlation_analysis_page import correlation_analysis_page
# from app_pages.sales_price_prediction_page import sales_price_prediction_page
from app_pages.project_hypothesis_page import project_hypothesis_page_body
from app_pages.ml_price_prediction import ml_price_prediction_page

# The app is instantiated here
app = MultiPage(app_name="Heritage Housing Sale Price Predictor")

# App pages are added here with .add_page()
app.add_page("Project Overview", summary_page_body)
app.add_page("Correlation Analysis", correlation_analysis_page)
# app.add_page("Sale Price Prediction", sales_price_prediction_page)
app.add_page("Hypothesis Validation", project_hypothesis_page_body)
app.add_page("Machine Learning Model", ml_price_prediction_page)

app.run()

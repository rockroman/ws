import streamlit as st
from app_pages.multipage import MultiPage
from app_pages.test_page import test


app = MultiPage(app_name="Heritage Housing Sale Price Predictor")

app.add_page("Test Page", test)

app.run()

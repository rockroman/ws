import streamlit as st

def predict_inherited_house_price(X_inherited, house_features, price_pipeline):

	# From inherited houses data, subset features related to this pipeline
	X_inherited_price = X_inherited.filter(house_features)

	# prediction

	price_prediction_inherited = price_pipeline.predict(X_inherited_price)
		
	this_price = price_prediction_inherited[0]
		
	return this_price
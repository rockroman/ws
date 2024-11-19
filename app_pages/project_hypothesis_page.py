import streamlit as st

def project_hypothesis_page_body():

    st.write("### Project Hypotheses and Validation")

     
    st.success(
       f"**First Hypothesis: Size Hypothesis** We hypothesize that larger properties tend to have higher absolute sale prices.\n"
        f"* **Hypothesis confirmed** The correlation study shows that features that capture the size of\
             a property are positively and moderately correlated with sale price.\n\n"
       
       f"**Second Hypothesis - Quality Hypothesis** We suspect that the overall quality of a house will significantly impact the sale price.\ 
       The higher the quality rating, the higher the expected sale/market price.\n"
        f"* **Hypothesis confirmed**  The correlation between sale price and overall quality\
             shows that hypothesis can not be proven wrong and that hypothesis is correct.\n\n"
       
       f"**Third hypothesis** We suspect that the overall condition of a house will influence the sale price.\
           Houses in better condition should command a higher price.\n"
        f"* **Hypothesis confirmed** The hypothesis was validated with correlations between the sale price and\
             the construction year or year of remodeling of the property. These features correlated positively and moderately with sale price\n"
    )

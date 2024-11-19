import streamlit as st

def project_hypothesis_page_body():
    st.write("### Project Hypotheses and Validation")

    st.success(
        f"**First Hypothesis: Size Hypothesis** We hypothesize that larger properties tend to have higher absolute sale prices.\n"
        f"* **Hypothesis confirmed:** The correlation study shows that features that capture the size of "
        f"a property are positively and moderately correlated with sale price.\n\n"
        
        f"**Second Hypothesis - Quality Hypothesis** We suspect that the overall quality of a house will significantly impact the sale price. "
        f"The higher the quality rating, the higher the expected sale/market price.\n"
        f"* **Hypothesis confirmed:** The correlation between sale price and overall quality "
        f"shows that hypothesis cannot be proven wrong and is likely correct.\n\n"
        
        f"**Third Hypothesis** We suspect that the overall condition of a house will influence the sale price. "
        f"Houses in better condition should command a higher price.\n"
        f"* **Hypothesis confirmed:** The hypothesis was validated with correlations between the sale price and "
        f"the construction year or year of remodeling of the property. These features correlated positively and moderately with sale price.\n"
    )

import streamlit as st


def summary_page_body():
    """
    Displays contents of the project summary page.
    """
    st.write("# Project Overview")

    st.info(
        "**Project Background**\n\n"
        "Our client has inherited four houses in Ames, Iowa, and asked for our "
        "assistance to maximize the sales price. To that end, a Machine Learning "
        "model and regression algorithms were created to provide the best possible "
        "price indication for the property inherited."
    )

    st.write("# Project Dataset")

    st.info(
        "**Dataset Source**\n\n"
        "The dataset is sourced from Kaggle, with approximately 1,500 records of "
        "housing sales in Ames, Iowa. Each record includes 23 features that describe "
        "the house profile. Features include information about the floor area, basement, "
        "and existence of a garage."
    )

    st.write("# Business Requirements")

    st.success(
        "**Project Objectives**\n\n"
        "The project has two primary objectives:\n"
        "* **BR1**: Discover and analyze how house attributes correlate with sales price "
        "supported by data visualizations to show the relationship.\n"
        "* **BR2**: Development of a prediction model to estimate the sale prices of the "
        "inherited houses and other properties in Ames, Iowa."
    )

    st.write("# Additional Information")

    st.write(
        "* For more information on this project, please visit the "
        "[README file](https://github.com/Werner-Staeblein/Project-5/blob/main/README.md)."
    )


# Call the function to display the content on the Streamlit page
summary_page_body()

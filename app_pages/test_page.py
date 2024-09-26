import streamlit as st

def test():
    st.write("# This is a test page to see the rendering when deployed")
    st.info(
        """This is just a simple text to prepare for testing the
        rendering and deployment of the app on Heroku. Deployment test done from the start."""
    )
    st.write("# Another Test Headline")
    st.success(
        """This is just a simple text to prepare for testing the
        rendering and deployment of the app on Heroku. Deployment test done from the start."""
    )


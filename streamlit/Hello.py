import streamlit as st

st.set_page_config(
    page_title="Model Stats for Titanic",
    page_icon="👋",
)

st.write("# Welcome to Titanic Model Stats!")

st.sidebar.write("Click on pages above for details.")

st.markdown(
    """
    
    This is developed using Streamlit. You can make yourself quick dashboards.
    
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    
    
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
        
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)
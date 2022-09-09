import pandas as pd
import numpy as np
import pandas_profiling
import streamlit as st
from streamlit_pandas_profiling import st_profile_report


@st.cache
def load_runs():
    dftrain = pd.read_csv("data/train.csv")
    return dftrain

st.title('EDA')

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_runs()
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

pr = data.profile_report()

st.title("Pandas Profiling in Streamlit")
st_profile_report(pr)

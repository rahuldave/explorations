import streamlit as st
import pandas as pd
import numpy as np

@st.cache
def load_runs():
    df = pd.read_csv("results.csv")
    df = df.drop("cm", axis=1)
    dftrain = df[df.data=='train'].drop("data", axis=1)
    dftest = df[df.data=='test'].drop("data", axis=1)
    return dftest

st.title('Compare Runs of Experiment')

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_runs()
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

st.dataframe(data)

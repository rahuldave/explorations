import streamlit as st
import pandas as pd
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode


@st.cache
def load_runs():
    df = pd.read_csv("results.csv")
    df = df.drop("cm", axis=1)
    dftrain = df[df.data=='train'].drop("data", axis=1)
    dftest = df[df.data=='test'].drop("data", axis=1)
    return dftrain, dftest

st.title('Compare Runs of Experiment')

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data_train, data_test = load_runs()
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

tab_test, tab_train = st.tabs(['Test', 'Train'])
#tab_train, tab_test = st.tabs(['Train', 'Test'])



with tab_test:
    dtr4 = data_test.round(4)
    st.header("Test")
    gb = GridOptionsBuilder.from_dataframe(dtr4)
    gb.configure_column("run_id",
                        cellRenderer=JsCode('''function(params) {return '<a target="_blank" href="/Model_Details?option='+params.value+'">'+params.value+'</a>'}'''),
                        width=300)
    grid_options = gb.build()
    a1 = AgGrid(dtr4, grid_options, allow_unsafe_jscode=True, theme="streamlit",  columns_auto_size_mode=2) 
    #, fit_columns_on_grid_load=True)


with tab_train:
    dte4 = data_train.round(4)
    st.header("Train")
    gbt = GridOptionsBuilder.from_dataframe(dte4)
    gbt.configure_column("run_id",
                        cellRenderer=JsCode('''function(params) {return '<a target="_blank" href="/Model_Details?option='+params.value+'">'+params.value+'</a>'}'''),
                        width=300)
    grid_optionst = gbt.build()
    a2 = AgGrid(dte4, grid_optionst, allow_unsafe_jscode=True, theme="streamlit",  columns_auto_size_mode=1)

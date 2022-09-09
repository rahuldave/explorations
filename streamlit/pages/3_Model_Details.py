import streamlit as st
import pandas as pd
import numpy as np
import joblib

hyperparams = ["n_est", "max_depth"]
metrics = ['roc_auc', 'average_precision', 'accuracy', 'precision', 'recall', 'f1']
metric_names = ['AUC', 'Av. Precision', "Accuracy", "Precision", "Recall", "F1"]
plots = ['cm', 'pr', 'roc']
plot_names = ['Confusion Matrix', 'Precision Recall Curve', 'ROC Curve']
# @st.cache
def load_runs():
    df = pd.read_csv("results.csv")
    df = df.drop("cm", axis=1)
    dftrain = df[df.data=='train'].drop("data", axis=1).set_index('run_id')
    dftest = df[df.data=='test'].drop("data", axis=1).set_index('run_id')
    return dftrain, dftest

def load_model(modelid):
    return joblib.load(modelid)
    

st.title('Run Details')

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data_train, data_test = load_runs()
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

qparam = st.experimental_get_query_params()
if "option" in qparam:
    option = qparam['option'][0]
else:
    option = st.selectbox(
         'Which run would you like to see?',
         list(data_train.index))

st.header(f"Run: {option}")

test_dict = dict(data_test.loc[option])
train_dict = dict(data_train.loc[option])

t1, t2 = st.tabs(['Metrics and Plots', 'Inference'])

with t1:
    for p in hyperparams:
        st.write(p, test_dict[p])

    st.subheader("Test Set")

    cols = st.columns(len(metrics))
    for i, m in enumerate(metrics):
        cols[i].metric(metric_names[i], round(test_dict[metrics[i]], 2))

    cols = st.columns(len(plots))
    for i, p in enumerate(plots):
        plotfile = f"plots/test_{option}_{p}.png"
        cols[i].image(plotfile, plot_names[i])

    st.subheader("Training Set")

    cols = st.columns(len(metrics))
    for i, m in enumerate(metrics):
        cols[i].metric(metric_names[i], round(train_dict[metrics[i]], 2))

    cols = st.columns(len(plots))
    for i, p in enumerate(plots):
        plotfile = f"plots/train_{option}_{p}.png"
        cols[i].image(plotfile, plot_names[i])

with t2:
    mappings = {'Sex': {'female': 0, 'male': 1},
     'Cabin': {'A': 0,
      'B': 1,
      'C': 2,
      'D': 3,
      'E': 4,
      'F': 5,
      'G': 6,
      'T': 7,
      np.nan: 8},
     'Embarked': {'C': 0, 'Q': 1, 'S': 2, np.nan: 3}}
    
    col1, col2 = st.columns(2)
    
    pclass_labels = ('1st Class', '2nd Class', '3rd Class')
    pclasses = (1, 2, 3)
    pclassmap = dict(zip(pclass_labels, pclasses))
    pclassval = col1.radio("Which Class was the passenger in?", pclass_labels)
    
    sex_labels = ('female', 'male')
    sexval = col2.radio("Sex of Passenger?", sex_labels)
    
    age = st.slider("Age", 0, 100, 40)
    fare = st.slider("Fare", 0, 750, 50)
    
    col1, col2 = st.columns(2)
    
    cabin_labels = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'Dont Know')
    cabins = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    cabinmap = dict(zip(cabin_labels, cabins))
    cabinval = col1.selectbox("Which Cabin Set was the passenger in?", cabin_labels)
    
    embark_labels = ('Cherbourg', 'Queenstown', 'Southampton', 'Dont Know')
    embarks = ('C', 'Q', 'S', np.nan)
    embarkmap = dict(zip(embark_labels, embarks))
    embarkval = col2.selectbox("Where did the passenger embark from?", embark_labels)
    
    family = st.slider("Family members in tow", 0, 20, 0)

    data_to_predict = dict(
        Pclass = pclassmap[pclassval],
        Sex = sexval,
        Age = age,
        Fare = fare,
        Cabin = cabinmap[cabinval],
        Embarked = embarkmap[embarkval],
        Family = family
    )
    for k in data_to_predict:
        if k in ['Sex', 'Embarked']:
            data_to_predict[k] = mappings[k][data_to_predict[k]]
                     
    rehydrated = load_model(f"models/{option}.joblib")
    inputX = np.array(list(data_to_predict.values())).reshape(1, -1)
    pred = rehydrated.predict(inputX)[0]
    proba = rehydrated.predict_proba(inputX)
    lifemap = {0: 'Death', 1: 'Survival'}
    st.write(f"We predict **{lifemap[pred]}**. (Probability of survival: {proba[0][1]}).")
                     
                     

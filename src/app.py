# from utils import db_connect
# engine = db_connect()

import streamlit as st
from pickle import load

st.set_page_config(page_title="MUSHROOM FINALPRO", layout="wide")
st.title("Predictor de setas comestibles y no-comestibles")

model = load(open("../models/decision_tree_classifier_default_42.sav", "rb"))
class_dict = {
    "0": "Comestible",
    "1": "No comestible"
}

st.sidebar.header("Opciones")
st.sidebar.title('1. Parametros cuantitativos')

val1 = st.sidebar.slider("cap-diameter", min_value = 0.0, max_value = 1.0, step = 0.1)
val2 = st.sidebar.slider("stem-height", min_value = 0.0, max_value = 1.0, step = 0.1)
val3 = st.sidebar.slider("stem-width", min_value = 0.0, max_value = 1.0, step = 0.1)

st.sidebar.title('2. Parametros cualitativos')
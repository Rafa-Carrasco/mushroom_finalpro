import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pickle import load
import json
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="MUSHROOM FINALPRO", layout="wide")

# counter 

if "counter" not in st.session_state:
    st.session_state.counter = 0
st.session_state.counter += 1
st.text(f"This page has run {st.session_state.counter} times.")
# st.button("Run it again")

# cargar el modelo

model = load(open("../models/decision_tree_classifier_default_42.sav", "rb"))
class_dict = {
    "0": "No Comestible",
    "1": "comestible"
}

# cargar data original para sacar categorias de variables

data = pd.read_csv('../data/processed/test_data.csv') 
categorical_col_final = ['cap-shape', 'gill-color', 'stem-surface', 'stem-color', 'veil-color','spore-print-color', 'season']

# Leer los mapeos desde el archivo .txt
with open('label_encoders.txt', 'r') as file:
    label_mappings = json.load(file)

# Reconstruir los LabelEncoders desde los mapeos
label_encoders = {}
for column, mapping in label_mappings.items():
    encoder = LabelEncoder()
    encoder.classes_ = mapping['classes']
    label_encoders[column] = encoder


# Crear una función para obtener categorías únicas
def get_unique_values(campo):
    return label_encoders[campo].classes_




# ---------------- SIDEBAR -----------------

st.sidebar.title("Opciones")
st.sidebar.header('1. Parametros cuantitativos')

val1 = st.sidebar.slider("cap-diameter", min_value = 0.0, max_value = 1.0, step = 0.1)
val2 = st.sidebar.slider("stem-height", min_value = 0.0, max_value = 1.0, step = 0.1)
val3 = st.sidebar.slider("stem-width", min_value = 0.0, max_value = 1.0, step = 0.1)

st.sidebar.header('2. Parametros cualitativos')

# Crear selectboxes para cada variable categórica
cap_shape = st.sidebar.selectbox("Cap Shape", get_unique_values("cap-shape"))
gill_color = st.sidebar.selectbox("Gill Color", get_unique_values("gill-color"))
stem_surface = st.sidebar.selectbox("Stem Surface", get_unique_values("stem-surface"))
stem_color = st.sidebar.selectbox("Stem Color", get_unique_values("stem-color"))
veil_color = st.sidebar.selectbox("Veil Color", get_unique_values("veil-color"))
spore_print_color = st.sidebar.selectbox("Spore Print Color", get_unique_values("spore-print-color"))
season = st.sidebar.selectbox("Season", get_unique_values("season"))


# --------------- MAIN BODY --------------------------------------

st.title("Clasificador de setas según su comestibilidad")
st.text('Intro sobre los datos : problemas iniciales')
st.image('/images/seta1.jpg')
if st.button("Predecir"):
   
   
    # # Transformar las entradas del usuario utilizando los LabelEncoders
    # cap_shape_enc = label_encoders['cap-shape'].transform([cap_shape])[0]
    # gill_color_enc = label_encoders['gill-color'].transform([gill_color])[0]
    # stem_surface_enc = label_encoders['stem-surface'].transform([stem_surface])[0]
    # stem_color_enc = label_encoders['stem-color'].transform([stem_color])[0]
    # veil_color_enc = label_encoders['veil-color'].transform([veil_color])[0]
    # spore_print_color_enc = label_encoders['spore-print-color'].transform([spore_print_color])[0]
    # season_enc = label_encoders['season'].transform([season])[0]
    # prediction = str(model.predict([[val1, val2, val3, cap_shape_enc, gill_color_enc, stem_surface_enc, stem_color_enc, veil_color_enc, spore_print_color_enc, season_enc]])[0])
    prediction = str(model.predict([[val1, val2, val3, cap_shape, gill_color, stem_surface, stem_color, veil_color, spore_print_color, season ]])[0])
    pred_class = class_dict[prediction]
    st.write("Predicción:", pred_class)
    st.success('AVISO: Aunque este modelo tiene un 98.7 % de acierto en sus predicciones, no es recomendable comer setas silvestres sin autentico conocimiento del medio ')
    st.balloons()
  
    # variables numericas: cap-diameter,stem-height,stem-width,
    # variables categoricas: cap-shape,gill-color,stem-surface,stem-color,veil-color,spore-print-color,season
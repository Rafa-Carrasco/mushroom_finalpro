import streamlit as st
import pandas as pd
import numpy as np
from pickle import load
import json
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="MUSHROOM FINALPRO by Rafa Carrasco", layout="wide")

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

# Leer los mapeos desde el archivo .json
with open('label_encoders.json', 'r') as file:
    label_mappings = json.load(file)

# Reconstruir los LabelEncoders desde los mapeos
label_encoders = {}
for column, mapping in label_mappings.items():
    encoder = LabelEncoder()
    encoder.classes_ = np.array(mapping['classes'])
    # encoder.classes_ = mapping['classes']
    label_encoders[column] = encoder


# Crear una función para obtener categorías únicas
def get_unique_values(campo):
    return label_encoders[campo].classes_



# ---------------- SIDEBAR -----------------

st.sidebar.title("Opciones")
st.sidebar.header('1. Parametros cuantitativos')

cap_d = st.sidebar.slider("Diametro del sombrero (cm)", min_value = 0.38, max_value = 62.9, step = 1.0)
stem_h = st.sidebar.slider("Altura del pie (cm)", min_value = 0.0, max_value = 33.9, step = 1.0)
stem_w = st.sidebar.slider("Ancho del pie (mm)", min_value = 10.0, max_value = 100.0, step = 1.0)

st.sidebar.header('2. Parametros cualitativos')

# Crear selectboxes para cada variable categórica
cap_shape = st.sidebar.selectbox("Forma del sombrero", get_unique_values("cap-shape"))
gill_color = st.sidebar.selectbox("Color de las láminas", get_unique_values("gill-color"))
stem_surface = st.sidebar.selectbox("Superficie del pie", get_unique_values("stem-surface"))
stem_color = st.sidebar.selectbox("Color del pie", get_unique_values("stem-color"))
veil_color = st.sidebar.selectbox("Color del velo", get_unique_values("veil-color"))
spore_print_color = st.sidebar.selectbox("Color de las esporas", get_unique_values("spore-print-color"))
season = st.sidebar.selectbox("Temporada", get_unique_values("season"))

# Transformar los valores seleccionados usando los LabelEncoders
cap_shape_enc = label_encoders['cap-shape'].transform([cap_shape])[0]
gill_color_enc = label_encoders['gill-color'].transform([gill_color])[0]
stem_surface_enc = label_encoders['stem-surface'].transform([stem_surface])[0]
stem_color_enc = label_encoders['stem-color'].transform([stem_color])[0]
veil_color_enc = label_encoders['veil-color'].transform([veil_color])[0]
spore_print_color_enc = label_encoders['spore-print-color'].transform([spore_print_color])[0]
season_enc = label_encoders['season'].transform([season])[0]

# --------------- MAIN BODY --------------------------------------

st.title("Clasificador de setas")
st.image('./images/seta1.jpg')
tab1, tab2, tab3, tab4 = st.tabs(["Tab 1", "Tab2", "Tab 3", "Tab 4"])
tab1.write("1. Sobre los datos : problemas iniciales")
tab2.write("2. Graficos simples y descubrimientos sobre el EDA")
tab3.write("3. Modelos de Machine Learning utilizados")
tab3.write("Los bosques aleatorios (Random Forest) son un poderoso modelo de machine learning que puede manejar tareas tanto de clasificación como de regresión. Funcionan mediante la combinación de múltiples árboles de decisión para mejorar la precisión y controlar el sobreajuste.")


tab4.write("4. Reflexiones y dificultades")
tab4.write("La reflexion despues de un proyecto es siempre muy enriquecedora y aporta enormemente al aprendizaje")
tab4.write("1. Importancia del dominio del problema: Trabajar con datos relacionados con la alimentación y la salud pública requiere un entendimiento profundo del dominio. Es crucial asegurarse de tener información precisa y actualizada sobre las características de las setas y los riesgos asociados.")
tab4.write("2. Preprocesamiento de datos: En este proyecto, el preprocesamiento de datos juega un papel crucial, especialmente en la limpieza de datos y la codificación de variables categóricas. Asegurarse de que los datos estén correctamente preparados y que las transformaciones como el encoding se realicen de manera adecuada es fundamental para el rendimiento del modelo.")
tab4.write("3. Selección y ajuste de modelo: Elegir el modelo adecuado y ajustar sus hiperparámetros son pasos críticos en cualquier proyecto de aprendizaje automático. En tu caso, el uso de un árbol de decisión es una opción válida, pero considerar otros modelos como Random Forests o Support Vector Machines podría ser beneficioso dependiendo de la complejidad y distribución de los datos.")
tab4.write("4. Interpretación de resultados: Interpretar los resultados del modelo es clave para entender su eficacia y posibles áreas de mejora. Esto incluye analizar métricas de rendimiento como precisión, recall y la matriz de confusión, así como explorar errores comunes cometidos por el modelo.")
tab4.write("5. Ética y responsabilidad: Dado que este modelo puede tener implicaciones directas en la salud y seguridad de las personas, es fundamental abordar cuestiones éticas y de responsabilidad. Esto incluye la transparencia en la interpretación de resultados, así como la educación sobre las limitaciones del modelo y la importancia de la consulta con expertos en setas antes de tomar decisiones basadas en predicciones.")
tab4.write("6. Mejoras futuras: Siempre hay espacio para mejorar un proyecto. Podrías considerar la expansión del conjunto de datos, explorar técnicas más avanzadas de modelado como el ensamblaje de modelos o incluso aplicar técnicas de explicabilidad del modelo para entender mejor las decisiones del mismo.")
tab4.write("En resumen, trabajar en un proyecto como este no solo implica desarrollar habilidades técnicas en aprendizaje automático, sino también ser consciente del contexto y las implicaciones prácticas de los resultados.")

# st.image('./images/seta2.jpg')
# st.image('./images/seta3.jpg')

if st.button("Predecir"):
  
    # prediction = str(model.predict([[cap_d, stem_h, stem_w, cap_shape, gill_color, stem_surface, stem_color, veil_color, spore_print_color, season ]])[0])  
    prediction = str(model.predict([[cap_d, stem_h, stem_w, cap_shape_enc, gill_color_enc, stem_surface_enc, stem_color_enc, veil_color_enc, spore_print_color_enc, season_enc ]])[0])
    pred_class = class_dict[prediction]
    st.write("Predicción:", pred_class)
    st.success('AVISO: Aunque este modelo tiene un 98.7 % de acierto en sus predicciones, no es recomendable comer setas silvestres sin autentico conocimiento del medio ')
    # st.balloons()
  
   

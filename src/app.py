import streamlit as st
import pandas as pd
import numpy as np
from pickle import load
import json
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="DATASETAS by Rafa Carrasco", layout="wide")

# counter 

if "counter" not in st.session_state:
    st.session_state.counter = 0
st.session_state.counter += 1
st.write(f"This page has run {st.session_state.counter} times.")
# st.button("Run it again")

# cargar el modelo
# model = load(open("../models/decision_tree_classifier_default_42.sav", "rb"))
# model = load(open("../models/randomforest_classifier_mejores parametros_42.sav", "rb"))
model = load(open("../models/KNN_default_42.sav", "rb"))
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

st.header("DATASETAS: Clasificador de setas con Machine Learning")
col1, col2 = st.columns([2,2])

with col1:
        st.image('./images/datasetas-logo_sm.jpg', width=400, use_column_width = 'auto')

with col2:
        st.write("Bienvenido al clasificador de setas. Esta herramienta te ayudará a predecir si una seta es comestible o no basada en ciertas características. Por favor, sigue los siguientes pasos:")

with st.expander("Instruciones"):
        st.write("Paso 1: Introduce los Parámetros Cuantitativos: En el lado izquierdo de la pantalla, encontrarás una barra lateral con las opciones de entrada. Mueve el deslizador de izquierda a derecha para ajustar los valores.")
        st.write("Paso 2: Introduce los Parámetros Cualitativos: Debajo de los deslizadores, verás varias cajas de selección (select boxes) para diferentes características cualitativas de la seta. Selecciona una opción para cada característica.")
        st.write("Paso 3: Realiza la Predicción: Una vez que hayas ingresado todos los parámetros, haz clic en el botón 'Predecir' que se encuentra en el centro de la página. La aplicación procesará la información y mostrará el resultado de la predicción en la pantalla.")
        st.write("Paso 4: Interpreta el Resultado: La aplicación te mostrará si la seta es 'Comestible' o 'No Comestible' basado en los parámetros que elegiste. Ten en cuenta el aviso.") 

with st.container(border=True):
        st.write("AVISO IMPORTANTE: Esta herramienta es educativa y no debe ser utilizada como única fuente para determinar la comestibilidad de setas. Siempre consulta a un experto en micología antes de consumir setas silvestres.")
with st.container():
        st.write("Esperamos que esta guía te sea de ayuda. ¡Disfruta usando el clasificador de setas!")
st.divider()

if st.button("Predecir"):
  
    prediction = str(model.predict([[cap_d, stem_h, stem_w, cap_shape_enc, gill_color_enc, stem_surface_enc, stem_color_enc, veil_color_enc, spore_print_color_enc, season_enc ]])[0])
    pred_class = class_dict[prediction]
    st.write("Predicción:", pred_class)
    st.success('AVISO: Aunque este modelo tiene un 98.7 % de acierto en sus predicciones, no es recomendable comer setas silvestres sin autentico conocimiento del medio ')
    # st.balloons()



# -------------------------------TABS --------------------------------
  
tab1, tab2, tab3 = st.tabs(["Sobre los datos y el EDA", "Modelos de ML", "Reflexiones y dificultades"])

tab1.write("1. Sobre los datos: problemas iniciales")
tab1.write('''
 - Variabilidad: Todas las variables muestran una variabilidad considerable, con desviaciones estándar altas en comparación con sus medias. 
           Esto sugiere que las características físicas de los hongos en el conjunto de datos varían ampliamente.
 - Valores Extremos: La presencia de valores mínimos de 0 en stem-height y stem-width podría indicar errores en los datos o valores faltantes. 
 - Distribución Asimétrica: Las diferencias significativas entre las medianas y las medias, especialmente en stem-width, 
           sugieren que la distribución de los datos podría tener valores atípicos.
Este análisis descriptivo nos da una base para entender la distribución y las características de las variables en el dataset de setas. 
           Es un primer paso crucial antes de proceder con análisis más complejos o modelos predictivos.           
''')
tab1.write('''
 - Algunas variables tienen valor claramente predominante: habitat, ring-type, veil-color, y algo menos en otras como cap-color
           Las 3 variables numericas tienen un numero importante de outliers. 
           Dado el caracter de los datos y a su contexto de alimentación y salud pública  (aunque haya muy poco valores atipicos, 
           estos podrian ser determinantes a la hora de su comestibilidad) y el alto numero de variables que participan, 
           parece lo mas prudente mantener los outliers
   ''')
tab1.write('''
Conclusion clave sobre los datos:  

 - Ninguna de las caracteristicas por si misma parece determinante de la clase, 
           pero algunos valores parecen ser mayor factor de determinación que otros, debido al mayor numero de casos. 
           Ejemplo: ring-type=f; habitat=t; veil-color=w; cap-color=n;  
 - Los datos apuntan a que la determinación procede de combinaciones de distintas variables con ciertos valores. 
           Por ejemplo; la combinación: spore-print-color=k/n + cap-surface=t + stem-surface=y tiene una alta probabilidad de ser venenosa (p)
           ''')

tab1.write("2. Graficos sobre los datos")
with tab1:
   st.image("./images/output.png", use_column_width = 'auto', caption ="Proporcion de comestibles y no comestibles")
with tab1:
   st.image("./images/output_vars.png", use_column_width = 'auto', caption ="Comestibilidad segun atributos")
with tab1:
   st.image("./images/output_rf_impor.png", use_column_width = 'auto', caption ="")   



tab2.write("MODELOS DE MACHINE LEARNING")
tab2.write('''
           Como comentamos al principio, nuestro sistema de clasificación estaría basado en los atributos de las setas visibles a simple vista.
Elegir el modelo adecuado y ajustar sus hiper parámetros son pasos críticos en cualquier proyecto de aprendizaje automático. 
Después del EDA sabemos que nuestros datos tienen distribuciones asimétricas, con valores atípicos y extremos, mucha variabilidad entre atributos, 
           y muchos valores faltantes. Esto nos sirve como indicación para saber qué algoritmo puede dar los resultados más óptimos. 
En este proyecto, varios algoritmos de ML han sido testados. 
El uso de un árbol de decisión ha sido correcto como primera opción, pero comparar y mejorar el resultado 
           con otros modelos como RFs y KNN ha sido esencial para confirmar las validez de sus predicciones
        ''')
tab2.write(''' 
        MODELO 1: DECISION TREE: organiza decisiones y sus posibles consecuencias en una estructura de árbol. 
           Cada nodo interno representa una prueba en una característica (por ejemplo, si una característica es menor o mayor que un valor específico), 
           cada rama representa el resultado de la prueba, y cada nodo hoja representa una etiqueta de clase (para clasificación) o un valor continuo (para regresión). 
           El camino desde la raíz del árbol hasta una hoja representa una serie de decisiones que conducen a una predicción.
           ''')
with tab2:
   st.image("./images/output_dt.png", use_column_width = 'auto')
tab2.write(''' 
           MODELO 2: RANDOM FOREST: Se basa en la construcción de múltiples árboles de decisión durante el entrenamiento y su combinación para mejorar la precisión 
           y controlar el sobreajuste. Cada árbol en el bosque se entrena con una muestra diferente del conjunto de datos y se utiliza un subconjunto aleatorio 
           de características en cada división del árbol. La predicción final del modelo se obtiene tomando la mayoría de los votos (en el caso de la clasificación). 
            ''')        
with tab2:
   st.image("./images/rf_trees.png", use_column_width = 'auto')
tab2.write(''' 
           MODELO 3: K-NEAREST NEIGHBOURS: el algoritmo asigna una clase a un dato nuevo basándose en la clase más frecuente 
           entre sus k vecinos más cercanos en el espacio de características, donde k es un número entero predefinido.
        ''')
with tab2:
   st.image("./images/output_knn_cm.png", use_column_width = 'auto')


tab3.write("3. Reflexiones y dificultades")
tab3.write("La reflexion despues de un proyecto es siempre muy enriquecedora. La metacognición (análisis del proceso de aprendizaje) es un proceso clave que aporta enormemente al aprendizaje. Algunos puntos para reflexionar son:")
tab3.write("1. Importancia del dominio del problema: Trabajar con datos relacionados con la alimentación y la salud pública requiere un entendimiento profundo del dominio. Es crucial asegurarse de tener información precisa y actualizada sobre las características de las setas y los riesgos asociados. En este caso, los datos vienen con la garantia de un equipo de investigadores de la UCI")
tab3.write("2. Preprocesamiento de datos: En este proyecto, el preprocesamiento de datos juega un papel crucial, especialmente en la limpieza de datos y la codificación de variables categóricas. Asegurarse de que los datos estén correctamente preparados y que las transformaciones como el encoding se realicen de manera adecuada es fundamental para el rendimiento del modelo.")
tab3.write("3. Selección y ajuste de modelo: Elegir el modelo adecuado y ajustar sus hiperparámetros son pasos críticos en cualquier proyecto de aprendizaje automático. En este proyecto, Naibe Bayes parecia la mejor solucion pero los reultados de la prueba han sido contundentes en su contra. El uso de un árbol de decisión ha sido una opción válida, pero comparar el resultado con otros modelos como RFs ha sido esencial para confirmar las validez de sus predicciones")
tab3.write("4. Interpretación de resultados: Interpretar los resultados del modelo es clave para entender su eficacia y posibles áreas de mejora. Esto incluye analizar métricas de rendimiento como precisión, recall y la matriz de confusión, así como explorar errores comunes cometidos por el modelo.")
tab3.write("5. Ética y responsabilidad: Dado que este modelo puede tener implicaciones directas en la salud y seguridad de las personas, es fundamental abordar cuestiones éticas y de responsabilidad. Esto incluye la transparencia en la interpretación de resultados, así como la educación sobre las limitaciones del modelo y la importancia de la consulta con expertos en setas antes de tomar decisiones basadas en predicciones.")
tab3.write("6. Mejoras futuras: Siempre hay espacio para mejorar un proyecto. Se podría considerar la expansión del conjunto de datos, explorar técnicas más avanzadas de modelado como el ensamblaje de modelos o incluso aplicar técnicas de explicabilidad del modelo (como visualizacion de los arboles de decision, SHAP, LIME, PDP), para entender mejor las decisiones del mismo.")
tab3.write("En resumen, trabajar en un proyecto como este no solo implica desarrollar habilidades técnicas en aprendizaje automático, sino también ser consciente del contexto y las implicaciones prácticas de los resultados.")


# st.image('./images/seta3.jpg')


   

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
model = load(open("../models/randomforest_classifier_mejores parametros_42.sav", "rb"))
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
st.image('./images/datasetas-logo_sm.jpg')
with st.container():
        st.write("Bienvenido al clasificador de setas. Esta herramienta te ayudará a predecir si una seta es comestible o no basada en ciertas características. Por favor, sigue los siguientes pasos:")

with st.expander("Instruciones"):
# with st.container():
        st.write("Paso 1: Introduce los Parámetros Cuantitativos: En el lado izquierdo de la pantalla, encontrarás una barra lateral con las opciones de entrada. Mueve el deslizador de izquierda a derecha para ajustar los valores.")
        st.write("Paso 2: Introduce los Parámetros Cualitativos: Debajo de los deslizadores, verás varias cajas de selección (select boxes) para diferentes características cualitativas de la seta. Selecciona una opción para cada característica.")
        st.write("Paso 3: Realiza la Predicción: Una vez que hayas ingresado todos los parámetros, haz clic en el botón 'Predecir' que se encuentra en el centro de la página. La aplicación procesará la información y mostrará el resultado de la predicción en la pantalla.")
        st.write("Paso 4: Interpreta el Resultado: La aplicación te mostrará si la seta es 'Comestible' o 'No Comestible' basado en los parámetros que elegiste. Ten en cuenta el aviso.") 

with st.container(border=True):
        st.write("AVISO IMPORTANTE: Esta herramienta es educativa y no debe ser utilizada como única fuente para determinar la comestibilidad de setas. Siempre consulta a un experto en micología antes de consumir setas silvestres.")
with st.container():
        st.write("Esperamos que esta guía te sea de ayuda. ¡Disfruta usando el clasificador de setas!")

if st.button("Predecir"):
  
    prediction = str(model.predict([[cap_d, stem_h, stem_w, cap_shape_enc, gill_color_enc, stem_surface_enc, stem_color_enc, veil_color_enc, spore_print_color_enc, season_enc ]])[0])
    pred_class = class_dict[prediction]
    st.write("Predicción:", pred_class)
    st.success('AVISO: Aunque este modelo tiene un 98.7 % de acierto en sus predicciones, no es recomendable comer setas silvestres sin autentico conocimiento del medio ')
    # st.balloons()



# -------------------------------TABS --------------------------------
  
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Sobre los datos", "Sobre el EDA", "Modelo 1: Naive Bayes", "Modelo 2: Decision Tree", "Modelo 3: RF", "Reflexiones y dificultades"])

tab1.write("1. Sobre los datos : problemas iniciales")

tab2.write("2. Graficos simples y descubrimientos sobre el EDA")


tab3.write("MODELO 1: NAIVE BAYES")


tab4.write("MODELO 2: DECISION TREE")
tab4.write("Usar un algoritmo de árbol de decisión (Decision Tree) en un proyecto de clasificación de setas tiene varias ventajas:")
tab4.write("1. Facilidad de interpretación y visualización: son fáciles de entender y visualizar. Las decisiones tomadas en cada nodo y las reglas resultantes son transparentes y lógicas, lo que facilita la interpretación de cómo el modelo llega a una predicción. Esto es especialmente útil en proyectos donde la explicación del modelo a usuarios no técnicos (por ejemplo, aficionados a las setas) es crucial.")
tab4.write("2. Manejo de datos categóricos y numéricos: Los DT pueden manejar tanto características categóricas como numéricas sin necesidad de preprocesamiento extenso. Esto resulta muy practico en este proyecto, donde las características categóricas (como forma del sombrero, color de las láminas, etc.) son predominantes.")
tab4.write("3. No requiere suposiciones sobre la distribución de los datos: A diferencia de algunos algoritmos que asumen una distribución específica de los datos, los DT no tienen estos requisitos. Esto los hace más flexibles y aplicables a una variedad de conjuntos de datos.")
tab4.write("4. Manejo de los datos faltantes: Los DT pueden manejar valores faltantes en las características. Aunque tener datos completos y limpios es ideal, los DT pueden dividirse utilizando variables disponibles durante el entrenamiento.")
tab4.write("5. Identificación de interacciones entre variables: Los DT pueden capturar interacciones complejas entre variables sin necesidad de especificarlas explícitamente. Esto es útil en la clasificación de setas, donde hemos visto que las interacciones entre diferentes características son claves para determinar la comestibilidad.")
tab4.write("6. Eficiencia computacional: Entrenar y predecir con DT suele ser más rápido en comparación con otros algoritmos más complejos como las redes neuronales. Esto tiene muchas ventajas en aplicaciones móviles, por ejemplo.")
tab4.write("7. Menor sensibilidad a los valores atípicos: Los DT son menos afectados por valores atípicos en comparación con algunos algoritmos de aprendizaje automático.")
tab4.write("En resumen, los DT son una herramienta poderosa y versátil que puede ofrecer ventajas significativas en la clasificación de setas, tanto en términos de rendimiento como de interpretabilidad y facilidad de uso.")

tab5.write("MODELO 3: RANDOM FOREST")
tab5.write("El modelo Random Forest (RF) puede manejar tareas tanto de clasificación como de regresión. Funcionan mediante la combinación de múltiples DT para mejorar la precisión y controlar el sobreajuste. Utilizarlo en este proyecto ofrece varias ventajas significativas:")
tab5.write("1. Mejora de la Precisión: Al promediar las predicciones de muchos árboles, RF tiende a ser más preciso que un solo árbol de decisión.")
tab5.write("2. Resistencia contra el Sobreajuste: A diferencia de los árboles de decisión simples, que pueden ser propensos a sobreajustarse a los datos de entrenamiento, RF reduce este riesgo al crear árboles a partir de diferentes subconjuntos de datos y promediar sus predicciones. Esto mejora la capacidad general del modelo para generalizar en datos nuevos.")
tab5.write("3. Manejo de Datos Perdidos: RF puede manejar valores faltantes de manera efectiva. Durante la construcción de árboles, puede estimar valores faltantes y mantener el rendimiento del modelo incluso cuando los datos de entrada no están completos.")
tab5.write("4. Importancia de las Características: RF puede proporcionar una medida de importancia para cada característica del conjunto de datos. Esto es particularmente útil para identificar qué variables tienen el mayor impacto en la predicción de la comestibilidad de las setas, lo cual puede ser útil para la interpretación del modelo y para futuros esfuerzos de recopilación de datos.")
tab5.write("5. Versatilidad y Flexibilidad: RF puede ser utilizado tanto para tareas de clasificación como de regresión. En el contexto de este proyecto, se utiliza para la clasificación, pero su flexibilidad es una ventaja en caso de que el proyecto evolucione y requiera diferentes tipos de análisis.")
tab5.write("6. Resistencia a los Valores Atípicos: Debido a que RF promedia los resultados de muchos árboles, es menos sensible a los valores atípicos que los modelos de árboles de decisión únicos. Esto puede resultar en predicciones más estables y robustas.")
tab5.write("7. Facilidad de Uso: Una vez configurado, RF requiere relativamente pocos parámetros para ajustar en comparación con otros modelos complejos, lo que facilita su implementación y optimización.")
tab5.write("8. Eficiencia Computacional: Aunque construir muchos árboles puede parecer computacionalmente costoso, RF puede ser paralelizado fácilmente, permitiendo que múltiples árboles sean entrenados simultáneamente, lo que mejora la eficiencia computacional.")
tab5.write("9. Interpretabilidad: Si bien RF no es tan interpretable como un solo árbol de decisión, aún proporciona un buen equilibrio entre interpretabilidad y rendimiento. Las importancias de características y las estructuras de árboles individuales pueden ser inspeccionadas para obtener información sobre el modelo.")
tab5.write("En resumen, RF ofrece un equilibrio robusto entre precisión, manejo de datos faltantes, resistencia al sobreajuste y facilidad de uso, lo que lo convierte en una opción excelente para la clasificación de setas.")


tab6.write("4. Reflexiones y dificultades")
tab6.write("La reflexion despues de un proyecto es siempre muy enriquecedora. La metacognición (análisis del proceso de aprendizaje) es un proceso clave que aporta enormemente al aprendizaje. Algunos puntos para reflexionar son:")
tab6.write("1. Importancia del dominio del problema: Trabajar con datos relacionados con la alimentación y la salud pública requiere un entendimiento profundo del dominio. Es crucial asegurarse de tener información precisa y actualizada sobre las características de las setas y los riesgos asociados. En este caso, los datos vienen con la garantia de un equipo de investigadores de la UCI")
tab6.write("2. Preprocesamiento de datos: En este proyecto, el preprocesamiento de datos juega un papel crucial, especialmente en la limpieza de datos y la codificación de variables categóricas. Asegurarse de que los datos estén correctamente preparados y que las transformaciones como el encoding se realicen de manera adecuada es fundamental para el rendimiento del modelo.")
tab6.write("3. Selección y ajuste de modelo: Elegir el modelo adecuado y ajustar sus hiperparámetros son pasos críticos en cualquier proyecto de aprendizaje automático. En este proyecto, Naibe Bayes parecia la mejor solucion pero los reultados de la prueba han sido contundentes en su contra. El uso de un árbol de decisión ha sido una opción válida, pero comparar el resultado con otros modelos como RFs ha sido esencial para confirmar las validez de sus predicciones")
tab6.write("4. Interpretación de resultados: Interpretar los resultados del modelo es clave para entender su eficacia y posibles áreas de mejora. Esto incluye analizar métricas de rendimiento como precisión, recall y la matriz de confusión, así como explorar errores comunes cometidos por el modelo.")
tab6.write("5. Ética y responsabilidad: Dado que este modelo puede tener implicaciones directas en la salud y seguridad de las personas, es fundamental abordar cuestiones éticas y de responsabilidad. Esto incluye la transparencia en la interpretación de resultados, así como la educación sobre las limitaciones del modelo y la importancia de la consulta con expertos en setas antes de tomar decisiones basadas en predicciones.")
tab6.write("6. Mejoras futuras: Siempre hay espacio para mejorar un proyecto. Se podría considerar la expansión del conjunto de datos, explorar técnicas más avanzadas de modelado como el ensamblaje de modelos o incluso aplicar técnicas de explicabilidad del modelo (como visualizacion de los arboles de decision, SHAP, LIME, PDP), para entender mejor las decisiones del mismo.")
tab6.write("En resumen, trabajar en un proyecto como este no solo implica desarrollar habilidades técnicas en aprendizaje automático, sino también ser consciente del contexto y las implicaciones prácticas de los resultados.")


# st.image('./images/seta3.jpg')


   

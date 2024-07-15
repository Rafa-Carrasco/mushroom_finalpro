# DATASETAS  

DATASETAS es un clasificador de setas según su comestibilidad. 

## IDEA INICIAL 

El objetivo de este proyecto es investigar si el uso de algoritmos de machine learning es útil y eficaz para predecir si un conjunto de setas es comestible o no corresponde a la tarea de clasificarlos en dos grupos—comestibles o venenosos—sobre la base de una regla de clasificación. 

# PROCESO DE TRABAJO

Se ha realizado un EDA completo, y los datos muestran: 

 - Gran Variabilidad: Todas las variables muestran una variabilidad considerable, con desviaciones estándar altas en comparación con sus medias. 
Esto sugiere que las características físicas de los hongos en el conjunto de datos varían ampliamente.

 - Distribución Asimétrica: Las diferencias significativas entre las medianas y las medias, especialmente en stem-width, sugieren que la distribución de los datos podría tener valores atípicos. 

 - Ninguna de las características por sí misma parece determinante de la clase, pero algunos valores parecen ser mayor factor de determinación que otros, debido al mayor número de casos. Ejemplo: ring-type=f; habitat=t; veil-color=w; cap-color=n;

 - Los datos apuntan a que la determinación procede de combinaciones de distintas variables con ciertos valores. Por ejemplo; la combinación: spore-print-color=k/n + cap-surface=t + stem-surface=y tiene una alta probabilidad de ser venenosa (p)

 - Hay un alto número de atributos presentes en el dataset original. Para optimizar los modelos , elegiremos los que mayor impacto tienen en la caracteristica predictiva: es decir, ser comestibles o no. Para eso utilizaremos diferentes tecnicas y graficos. 

 - Después de hacer del EDA concluimos en que los atributos más relevantes son: 

1. Parámetros cuantitativos: Diámetro del sombrero (cm), Altura del pie (cm) y Ancho del pie (mm)
2. Parámetros cualitativos: Forma del sombrero, Color de las láminas, Superficie del pie, Color del pie, Color del velo, Color de las esporas, Temporada

Modelos de ML

 - El sistema de clasificación esta basado en los atributos de las setas visibles a simple vista.
 - Elegir el modelo adecuado y ajustar sus hiper parámetros son pasos críticos en cualquier proyecto de aprendizaje automático. 
 - Después del EDA sabemos que nuestros datos tienen distribuciones asimétricas, con valores atípicos y extremos y mucha variabilidad entre atributos, lo que nos sirve como indicación para saber qué algoritmo puede dar los resultados más óptimos. 
 - En este proyecto, varios algoritmos de ML han sido testados: Naive Bayes, Decision Tree, Random Foresy Y K-Nearest Neighbors.   
 - El uso de un árbol de decisión ha sido correcto como primera opción, pero comparar y mejorar el resultado con otros modelos como RFs y KNN ha sido esencial para confirmar las validez de sus predicciones


# BIBLIOGRAFIA Y RECURSOS 

El dataset original ha sido descargado de https://mushroom.mathematik.uni-marburg.de/ y creado por Dennis Wagner , Dominik Heider y Georges Hattab 
La informacion del mismo ha sido extraido mediante NLP del llibro de Patrick Hardin. "Mushrooms & Toadstools. Collins, 2012"

M. S. Morshed, F. Bin Ashraf, M. U. Islam and M. S. R. Shafi, "Predicting Mushroom Edibility with Effective Classification and Efficient Feature Selection Techniques," 2023 3rd International Conference on Robotics, Electrical and Signal Processing Techniques (ICREST), Dhaka, Bangladesh, 2023, pp. 1-5, doi: 10.1109/ICREST57604.2023.10070049. https://ieeexplore.ieee.org/document/10070049

Wagner, D., Heider, D. & Hattab, G. Mushroom data creation, curation, and simulation to support classification tasks. Sci Rep 11, 8134 (2021). https://doi.org/10.1038/s41598-021-87602-3


# ESTRUCTURA

El proyecto está organizado de la siguiente manera:

app.py - El script principal de Python que ejecuta el proyecto.
explore.py - jupiter notebook con todo le proceso del EDA y los diferentes modelos.
requirements.txt - Este archivo contiene la lista de paquetes de Python necesarios.
models/ - Este directorio contiene los diferentes modelos testados: NB, DT, RF y KNN
data/ - Este directorio contiene los siguientes subdirectorios:
interim/ - la base dedatos SQL con los datos limpios despues del EDA (sin escalar ni codificar)
processed/ - Para los datos finales a utilizar para el modelado.
raw/ - Dataset original sin ningún procesamiento. 


## Configuración

**Prerrequisitos**

Asegúrate de tener Python 3.11+ instalado en tu máquina. También necesitarás pip para instalar los paquetes de Python.

**Instalación**

Clona el repositorio del proyecto en tu máquina local.
Navega hasta el directorio del proyecto e instala los paquetes de Python requeridos:

```bash
pip install -r requirements.txt
```

## Ejecutando la Aplicación

Para ejecutar la aplicación, ejecuta el script app.py desde la raíz del directorio del proyecto:

```bash
streamlit run app.py
```

# sobre la plantilla 

Esta plantilla fue construida como parte del [Data Science and Machine Learning Bootcamp](https://4geeksacademy.com/us/coding-bootcamps/datascience-machine-learning) de 4Geeks Academy por [Alejandro Sanchez](https://twitter.com/alesanchezr) y muchos otros contribuyentes. Descubre más sobre [los programas BootCamp de 4Geeks Academy](https://4geeksacademy.com/us/programs) aquí.

Otras plantillas y recursos como este se pueden encontrar en la página de GitHub de la escuela.
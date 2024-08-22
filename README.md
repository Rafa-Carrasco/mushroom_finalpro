# DATASETAS

DATASETAS is a mushroom classifier based on edibility.

## INITIAL IDEA

The goal of this project is to investigate whether using machine learning algorithms is useful and effective for predicting if a set of mushrooms is edible or not. The task is to classify them into two groups—edible or poisonous—based on a classification rule.

## WORK PROCESS

A complete EDA (Exploratory Data Analysis) has been performed, and the data show:

- **High Variability:** All variables show considerable variability, with high standard deviations compared to their means. This suggests that the physical characteristics of the mushrooms in the dataset vary widely.

- **Asymmetric Distribution:** Significant differences between the medians and means, especially in stem-width, suggest that the data distribution might have outliers.

- **No Single Determinant Feature:** None of the features alone seems decisive for the class, but some values appear to be more determinant than others due to the higher number of cases. Examples: ring-type=f; habitat=t; veil-color=w; cap-color=n.

- **Combination of Variables:** The classification is likely determined by combinations of different variables with specific values. For example, the combination: spore-print-color=k/n + cap-surface=t + stem-surface=y has a high probability of being poisonous (p).

- **Feature Selection:** There is a high number of attributes in the original dataset. To optimize the models, we will choose those with the greatest impact on the predictive feature: edibility or not. Various techniques and charts will be used for this purpose.

- **Conclusion after EDA:** The most relevant attributes are:

  1. **Quantitative Parameters:** Cap diameter (cm), stem height (cm), and stem width (mm).
  2. **Qualitative Parameters:** Cap shape, gill color, stem surface, stem color, veil color, spore color, season.

## ML Models

- The classification system is based on the visible attributes of mushrooms.
- Choosing the right model and tuning its hyperparameters are critical steps in any machine learning project.
- After the EDA, we know our data has asymmetric distributions, with outliers and extremes, and a lot of variability between attributes, which helps us understand which algorithm might give the most optimal results.
- In this project, several ML algorithms have been tested: Naive Bayes, Decision Tree, Random Forest, and K-Nearest Neighbors.
- Using a decision tree was a good initial option, but comparing and improving the results with other models like RF and KNN was essential to confirm the validity of the predictions.

## BIBLIOGRAPHY AND RESOURCES

The original dataset was downloaded from https://mushroom.mathematik.uni-marburg.de/ and created by Dennis Wagner, Dominik Heider, and Georges Hattab. The information was extracted using NLP from Patrick Hardin's book "Mushrooms & Toadstools. Collins, 2012."

- M. S. Morshed, F. Bin Ashraf, M. U. Islam, and M. S. R. Shafi, "Predicting Mushroom Edibility with Effective Classification and Efficient Feature Selection Techniques," 2023 3rd International Conference on Robotics, Electrical and Signal Processing Techniques (ICREST), Dhaka, Bangladesh, 2023, pp. 1-5, doi: 10.1109/ICREST57604.2023.10070049. https://ieeexplore.ieee.org/document/10070049

- Wagner, D., Heider, D., & Hattab, G. Mushroom data creation, curation, and simulation to support classification tasks. Sci Rep 11, 8134 (2021). https://doi.org/10.1038/s41598-021-87602-3

## STRUCTURE

The project is organized as follows:

- **app.py** - The main Python script that runs the project.
- **explore.py** - Jupyter notebook with the entire EDA process and the different models.
- **requirements.txt** - This file contains the list of required Python packages.
- **models/** - This directory contains the different tested models: NB, DT, RF, and KNN.
- **data/** - This directory contains the following subdirectories:
  - **interim/** - The SQL database with cleaned data after the EDA (not scaled or encoded).
  - **processed/** - For the final data to be used for modeling.
  - **raw/** - Original dataset without any processing.

## Configuration

**Prerequisites**

Make sure you have Python 3.11+ installed on your machine. You will also need pip to install the Python packages.

**Installation**

Clone the project repository to your local machine. Navigate to the project directory and install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Application

To run the application, execute the app.py script from the root directory of the project:

```bash
streamlit run app.py
```

## About the Template

This template was built as part of the [Data Science and Machine Learning Bootcamp](https://4geeksacademy.com/us/coding-bootcamps/datascience-machine-learning) by 4Geeks Academy by [Alejandro Sanchez](https://twitter.com/alesanchezr) and many other contributors. Discover more about [4Geeks Academy BootCamp programs](https://4geeksacademy.com/us/programs) here.

Other templates and resources like this can be found on the school's GitHub page.

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Page confguration
st.set_page_config(
    page_title='Simple Prediction App',
    layout='wide',
    initial_sidebar_state='expanded'
)

# App Title
st.title('Simple Flower Prediction')

# Load Dataset
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/iris.csv')

# Input widgets
st.sidebar.subheader('Input Features')
sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.8)
sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.1)
petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 3.8)
petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 1.2)

# Split X and y
X = df.drop(columns = 'Species')
y = df.Species

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building
rf = RandomForestClassifier(max_depth=2, max_features=4, n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Apply model to make prediction
y_pred = rf.predict([[sepal_length, sepal_width, petal_length, petal_width]])

# Print EDA
st.subheader('Brief EDA')
st.write('The data is grouped by the class and the variable mean is computed for each class')
groupby_species_mean = df.groupby('Species').mean()
st.write(groupby_species_mean)
st.line_chart(groupby_species_mean.T)

# Print inout feature
input_feature = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=
                             ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
st.write(input_feature)

# Print prediction output
st.subheader('Output')
st.metric('Predicted Class', y_pred[0], '')
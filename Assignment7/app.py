import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import os

model = joblib.load("Assignment7/iris_model.pkl")


st.title("🌸 Iris Species Prediction App")
st.write("Enter flower measurements to predict the species.")


st.sidebar.header("🌿 Input Flower Measurements")

st.sidebar.markdown("You can either type the value or use the slider.")

sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_length_input = st.sidebar.number_input("Or type Sepal Length", value=sepal_length)

sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.0)
sepal_width_input = st.sidebar.number_input("Or type Sepal Width", value=sepal_width)

petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 4.0)
petal_length_input = st.sidebar.number_input("Or type Petal Length", value=petal_length)

petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.0)
petal_width_input = st.sidebar.number_input("Or type Petal Width", value=petal_width)


input_df = pd.DataFrame([[sepal_length_input, sepal_width_input, petal_length_input, petal_width_input]],
                        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])


if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"🌼 Predicted Species: **{prediction.capitalize()}**")

    
    image_path = f"Assignment7/images/{prediction.lower()}.jpg"
    if os.path.exists(image_path):
       st.image(Image.open(image_path), caption=f"{prediction.capitalize()} Flower", use_container_width=True)
    else:
        st.warning("Image not found for the predicted species.")

st.markdown("---")
if st.checkbox("📊 Show sample data"):
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    st.dataframe(df.head())

if st.checkbox("📈 Show visualization"):
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    fig, ax = plt.subplots()
    sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=df, ax=ax)
    st.pyplot(fig)

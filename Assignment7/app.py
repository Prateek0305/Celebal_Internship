import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os


model = joblib.load("Assignment7/iris_model.pkl")


st.set_page_config(page_title="Iris Species Predictor 🌸", layout="wide")
st.title("🌸 Iris Species Prediction App")
st.markdown("Enter flower measurements to predict the species using a trained ML model.")


tab1, tab2, tab3 = st.tabs(["📥 Prediction", "📈 Visualization", "📑 Sample Data"])


with tab1:
    st.sidebar.header("🌿 Input Flower Features")
    
    st.sidebar.markdown("Use sliders or number boxes below")


    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

    input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                            columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])

    st.subheader("📊 Your Input")
    st.dataframe(input_df, use_container_width=True)

    if st.button("🔍 Predict"):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        st.success(f"🌼 **Predicted Species:** `{prediction.capitalize()}`")

  
        st.subheader("🔢 Prediction Probabilities")
        proba_df = pd.DataFrame({
            "Species": model.classes_,
            "Probability": proba
        })
        st.bar_chart(proba_df.set_index("Species"))

      
        image_path = f"Assignment7/images/{prediction.lower()}.jpg"
        if os.path.exists(image_path):
            st.image(image_path, caption=f"{prediction.capitalize()} flower", use_container_width=True)
        else:
            st.warning("🌸 Flower image not found.")

 
    with st.expander("📚 About this App"):
        st.markdown("""
        - 🔍 **Model**: Logistic Regression  
        - 📂 **Dataset**: Iris (150 samples)  
        - 🎯 **Accuracy**: ~97%  
        - 🌱 **Features**: Sepal & Petal lengths and widths  
        """)


with tab2:
    st.subheader("📈 Sepal Length vs Sepal Width")
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    fig, ax = plt.subplots()
    sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=df, ax=ax)
    st.pyplot(fig)


with tab3:
    st.subheader("📄 Sample of Iris Dataset")
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    st.dataframe(df.head(20), use_container_width=True)

    if st.button("🎲 Show Random Sample"):
        random_row = df.sample(1).reset_index(drop=True)
        st.write(random_row)


st.markdown("---")
st.markdown("<p style='text-align:center'>Made with ❤️ by Prateek | Powered by Streamlit</p>", unsafe_allow_html=True)

# ⛽ MPG Prediction Web App

Predict fuel efficiency of a car and explore model insights using an interactive machine learning app built with Streamlit.

---

## 📌 Overview

This web application predicts the **Miles Per Gallon (MPG)** of a car based on user-input specifications like horsepower, weight, displacement, and more. It uses a trained **Random Forest Regressor** model and provides insights through various visualizations.

---

## 📂 Project Structure

```
Assignment7/
├── app.py
├── mpg_model.pkl
├── project_video.gif
└── README.md
```

---

## 🚀 Features

- 🎯 Predicts car fuel efficiency (MPG) using machine learning  
- 🧾 Sidebar to input car specs like horsepower, weight, etc.  
- 📊 Visualizations: Feature importance, actual vs predicted MPG, PDP  
- 🎨 Aesthetic and interactive Streamlit UI  
- 💡 Trained on seaborn's `mpg` dataset  

---

## 📽️ Sample Prediction Demo

![Sample Prediction](Assignment7/project_video.gif)

---

## 🔧 Technologies Used

- Python  
- Streamlit  
- Scikit-learn  
- Pandas, Matplotlib, Seaborn  
- Joblib  

---

## 📊 Input Parameters

- `Cylinders` - Number of engine cylinders  
- `Displacement` - Engine displacement (in cubic inches)  
- `Horsepower` - Engine horsepower  
- `Weight` - Vehicle weight (lbs)  
- `Acceleration` - Time taken to accelerate from 0 to 60 mph  
- `Model Year` - Year the car model was released  
- `Origin` - Manufacturing origin (USA, Europe, Japan)  
- `Car Name` - Vehicle model name  

---

## 🧠 Model Insights

- **Feature Importance** plot shows the most influential features on MPG  
- **Actual vs Predicted** scatter plot gives a sense of model accuracy  
- **Partial Dependence Plot** helps visualize how individual features affect predictions  

---

## 📦 How to Run

1. Clone this repo:
   ```
   git clone <repo-url>
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the app:
   ```
   streamlit run Assignment7/app.py
   ```

---

## 👨‍💻 Author

Made with ❤️ by **Prateek Agrawal**

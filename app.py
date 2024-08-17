import streamlit as st
import pandas as pd
import pickle
import shap
import xgboost as xgb

# HTML and title
html_temp = """
<div style="background-color:yellow;padding:1.5px">
<h1 style="color:black;text-align:center;">Used Car Price Prediction</h1>
</div><br>"""
st.markdown(html_temp, unsafe_allow_html=True)

st.write("\n\n"*2)

# Load the model and explainer
filename = 'Auto_Price_Pred_Model'
model = pickle.load(open(filename, 'rb'))

# Load training data to fit the explainer
# Replace 'train_data.csv' with your actual training data file
train_data = pd.read_csv('train_data.csv')  # Ensure this file has the same features as used for prediction
X_train = train_data.drop('target', axis=1)  # Replace 'target' with the actual target column name

explainer = shap.TreeExplainer(model)

with st.sidebar:
    st.subheader('Car Specs to Predict Price')

    make_model = st.sidebar.selectbox("Model Selection", ("Audi A3", "Audi A1", "Opel Insignia", "Opel Astra", "Opel Corsa", "Renault Clio", "Renault Espace", "Renault Duster"))
    hp_kW = st.sidebar.number_input("Horse Power:", min_value=40, max_value=294, value=120, step=5)
    age = st.sidebar.number_input("Age:", min_value=0, max_value=3, value=0, step=1)
    km = st.sidebar.number_input("km:", min_value=0, max_value=317000, value=10000, step=5000)
    Gears = st.sidebar.number_input("Gears:", min_value=5, max_value=8, value=5, step=1)
    Gearing_Type = st.sidebar.radio("Gearing Type", ("Manual", "Automatic", "Semi-automatic"))

    my_dict = {"make_model": make_model, "hp_kW": hp_kW, "age": age, "km": km, "Gears": Gears, "Gearing_Type": Gearing_Type}
    df = pd.DataFrame.from_dict([my_dict])

    cols = {
        "make_model": "Car Model",
        "hp_kW": "Horse Power",
        "age": "Age",
        "km": "km Traveled",
        "Gears": "Gears",
        "Gearing_Type": "Gearing Type"
    }

    df_show = df.copy()
    df_show.rename(columns=cols, inplace=True)
    st.write("Selected Specs: \n")
    st.table(df_show)

    if st.button("Predict"):
        pred = model.predict(df)
        col1, col2 = st.columns(2)
        col1.write("The estimated value of car price is €")
        col2.write(pred[0].astype(int))

        # SHAP explanation
        st.subheader("SHAP Explanation")

        # Ensure input data format is the same as the training data
        # Adjust feature names if needed
        feature_names = X_train.columns
        df_for_shap = df[feature_names]

        # Generate SHAP values
        shap_values = explainer.shap_values(df_for_shap)

        # Summary plot
        st.subheader("SHAP Summary Plot")
        st_shap(shap.summary_plot(shap_values, df_for_shap, show=False))

        # Force plot
        st.subheader("SHAP Force Plot")
        st_shap(shap.force_plot(explainer.expected_value, shap_values[0], df_for_shap, show=False))

        # Decision plot
        st.subheader("SHAP Decision Plot")
        st_shap(shap.decision_plot(explainer.expected_value, shap_values[0], df_for_shap, feature_names=feature_names))

st.write("\n\n")

import streamlit as st
import os
import pickle
import numpy as np
import pandas as pd
import re
from PIL import Image
import shap
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ----- Setting the page configurations
st.set_page_config(page_title="Retail Sales Prediction App", page_icon=":heavy_dollar_sign:", layout="wide", initial_sidebar_state="auto")

# Setting the page title
st.title("Retail Sales Prediction with Machine Learning")

# ---- Importing and creating other key elements items
# Function to import the Machine Learning toolkit
@st.cache(allow_output_mutation=True)
def load_ml_toolkit(relative_path):
    """
    This function loads the ML items/toolkit into this file by taking the relative path to the ML items/toolkit.
    Args:
        relative_path (string, optional): It receives the file path to the ML toolkit for loading.
    Returns:
        file: It returns the pickle file (which contains the Machine Learning items in this case).
    """
    with open(relative_path, "rb") as file:
        loaded_object = pickle.load(file)
    return loaded_object

# Function to load the dataset
@st.cache()
def load_data(relative_path):
    """
    This function is used to load the DataFrame into the current file.
    Args:
        relative_path (string): The relative path to the DataFrame to be loaded.
    Returns:
        DataFrame: Returns the DataFrame at the path provided.
    """
    merged_data = pd.read_csv(relative_path, index_col=0)
    merged_data["onpromotion"] = merged_data["onpromotion"].apply(int)
    merged_data["store_nbr"] = merged_data["store_nbr"].apply(int)
    merged_data["sales_date"] = pd.to_datetime(merged_data["sales_date"]).dt.date
    return merged_data

# Function to get date features from the inputs
@st.cache()
def getDateFeatures(df, date):
    """
    Function to extract date features from the inputs provided.
    Args:
        df (DataFrame): This is the DataFrame of the inputs to be processed for prediction.
        date (str): This is a string of the date column in the DataFrame which is to be processed.
    Returns:
        DataFrame: The function returns a revised DataFrame which contains the original DataFrame with the date features.
    """
    df["date"] = pd.to_datetime(df[date])
    df["day_of_week"] = df["date"].dt.dayofweek.astype(int)
    df["day_of_month"] = df["date"].dt.day.astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear.astype(int)
    df["is_weekend"] = np.where(df["day_of_week"] > 4, 1, 0).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["year"] = df["date"].dt.year.astype(int)
    df = df.drop(columns="date")
    return df

# ----- Loading the key components
# Loading the base dataframe
rpath = r"streamlit_src\merged_data.csv"
merged_data = load_data(rpath)

# Loading the toolkit
loaded_toolkit = load_ml_toolkit(r"streamlit_src\ML_toolkit")
if "results" not in st.session_state:
    st.session_state["results"] = []

# Instantiating the elements of the Machine Learning Toolkit
mm_scaler = loaded_toolkit["scaler"]
dt_model = loaded_toolkit["model"]
oh_encoder = loaded_toolkit["encoder"]

# SHAP explainer
explainer = shap.Explainer(dt_model, mm_scaler.transform(merged_data.drop(columns="sales")))

# ---- App sections ----
# Defining the base containers/ main sections of the app
header = st.container()
dataset = st.container()
features_and_output = st.container()

# Icon for the page
image = Image.open(r"streamlit_src\AI-Sales-Prediction.jpg")

# Instantiating the form to receive inputs from the user
form = st.form(key="information", clear_on_submit=True)

# Structuring the header section
with header:
    header.write("This app is built on a machine learning model to predict the sales of a retail company based on given variables for which you will make inputs (see the input section below). The model was trained based on the Corporation Favorita Retail Sales dataset.")
    header.image(image)
    header.markdown("*Source: [Artificial Intelligence for Sales Prediction](https://www.mydatamodels.com/solutions/artificial-intelligence-for-sales-prediction/)*")
    header.write("---")

# Designing the sidebar
st.sidebar.header("Information on Columns")
st.sidebar.markdown(""" 
                    - **store_nbr** identifies the store at which the products are sold.
                    - **family** identifies the type of product sold.
                    - **sales** is the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units(1.5 kg of cheese, for instance, as opposed to 1 bag of chips).
                    - **onpromotion** gives the total number of items in a product family that were being promoted at a store at a given date.
                    - **sales_date** is the date on which a transaction / sale was made
                    - **city** is the city in which the store is located
                    - **state** is the state in which the store is located
                    - **store_type** is the type of store, based on Corporation Favorita's own type system
                    - **cluster** is a grouping of similar stores.
                    - **oil_price** is the daily oil price
                    - **holiday_type** indicates whether the day was a holiday, event day, or a workday
                    - **locale** indicates whether the holiday was local, national or regional.
                    - **transferred** indicates whether the day was a transferred holiday or not.
                    """)

# Structuring the dataset section
with dataset:
    if dataset.checkbox("Preview the dataset"):
        dataset.write(merged_data.head())
        dataset.write("For more information on the columns, kindly take a look at the sidebar")
    dataset.write("---")

# Defining the list of expected variables
expected_inputs = ["sales_date", "family", "store_nbr", "store_type", "cluster", "city", "state", "onpromotion", "oil_price", "holiday_type", "locale", "transferred"]

# List of features to encode
categoricals = ["family", "city", "state", "store_type", "holiday_type", "locale"]

# List of features to scale
cols_to_scale = ["onpromotion"]

# Structuring the features and output section
with features_and_output:
    features_and_output.subheader("Inputs")
    features_and_output.write("This section captures your input to be used in predictions")

    left_col, mid_col, right_col = features_and_output.columns(3)

    # Designing the input section of the app
    with form:
        left_col.markdown("***Product and Transaction Data***")
        sales_date = left_col.date_input("Select a date:", min_value=merged_data["sales_date"].min())
        family = left_col.selectbox("Product family:", options=sorted(list(merged_data["family"].unique())))
        onpromotion = left_col.number_input("Number of products on promotion:", min_value=merged_data["onpromotion"].min(), value=merged_data["onpromotion"].min())
        city = left_col.selectbox("City:", options=sorted(set(merged_data["city"])))

        mid_col.markdown("***Store Data***")
        store_nbr = mid_col.selectbox("Store number:", options=sorted(set(merged_data["store_nbr"])))
        store_type = mid_col.radio("Store type:", options=sorted(set(merged_data["store_type"])), horizontal=True)
        cluster = mid_col.select_slider("Store cluster:", options=sorted(set(merged_data["cluster"])))
        state = mid_col.selectbox("State:", options=sorted(set(merged_data["state"])))

        right_col.markdown("***Data on External Factors***")
        oil_price = right_col.number_input("Oil price:", min_value=merged_data["oil_price"].min(), value=merged_data["oil_price"].min())
        if right_col.checkbox("Is it a holiday? (Check if holiday)"):
            holiday_type = right_col.selectbox("Holiday type:", options=sorted(set(merged_data["holiday_type"])))
            locale = right_col.selectbox("Locale:", options=sorted(set(merged_data["locale"])))
        else:
            holiday_type = "Work Day"
            locale = "National"

        # Submit button
        submitted = form.form_submit_button(label="Submit")

if submitted:
    with features_and_output:
        # Inputs formatting
        input_dict = {
            "sales_date": [sales_date],
            "family": [family],
            "store_nbr": [store_nbr],
            "store_type": [store_type],
            "cluster": [cluster],
            "city": [city],
            "state": [state],
            "onpromotion": [onpromotion],
            "oil_price": [oil_price],
            "holiday_type": [holiday_type],
            "locale": [locale],
            "transferred": [1 if right_col.checkbox("Is it a holiday? (Check if holiday)") else 0]
        }

        input_df = pd.DataFrame(input_dict)
        input_df = getDateFeatures(input_df, "sales_date")

        # Encode and scale
        input_df_encoded = oh_encoder.transform(input_df[categoricals])
        input_df_scaled = mm_scaler.transform(input_df[cols_to_scale])

        input_df_final = pd.concat([pd.DataFrame(input_df_encoded), pd.DataFrame(input_df_scaled)], axis=1)

        # Prediction
        prediction = dt_model.predict(input_df_final)
        st.subheader(f"Predicted Sales: ${prediction[0]:,.2f}")

        # SHAP values and visualization
        shap_values = explainer(input_df_final)
        st_shap(shap.summary_plot(shap_values, input_df_final, plot_type="bar"), height=600)

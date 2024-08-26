import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Jose's Portfolio", page_icon=":🧉:", layout="wide")

# Intro
with st.container():
    st.subheader("Hi, I am Jose")
    st.title("A Quantitative Economics Student and Aspiring Data Scientist")
    st.subheader("Links:")
    st.write(
        """
        [GitHub](https://github.com/jsd115) | [LinkedIn](https://www.linkedin.com/in/jose-de-los-rios)
        """
    )
    st.write(
        "I use tools such as Python, R Programming Language and SQL to create projects that can hopefully communicate interesting insights from data. You can find some of my projects below. Outside of this portfolio, I am working on many other projects, such as other ML models and financial analysis tools. I am always looking for new work and opportunities to learn! Feel free to reach out to me on LinkedIn or GitHub."
    )

# Project Summaries
with st.container():
    st.write("---")
    st.header("Projects:")
    st.write("##")
    st.write("---")
    st.header("Project 1")
    st.write(
            """
            The first project I wanted to highlight is programmed in Python. It is a web app created using the Dash library. The app features many other commonly used libraries in the data science space such as pandas and plotly. More about this project can be accessed on the Dash App Sales Analysis tab.
            """
        )
    st.write("---")
    st.header("Project 2")
    st.write(
            """
            The second project is created using the Sklearn ML framework. This is a stock price predictor yielding high accuracy. It comebines Sklearn's DecisionTreeRegression algorithm with knowledge of technical indicators. More about this project can be accessed on the Sklearn Stock Analysis tab. 
            """
        )
    st.write("---")
    st.header("Project 3")
    st.write(
            """
             Third, I leveraged the XGBoost library, another common python framework for machine learning, to create a credit card fraud detection model based off of a popular Kaggle dataset. This project combined the utility of sklearn in model selection and hyperparameter tuning to output a very accurate model. More information about this can be seen in the XGBoost  Card Fraud Detection tab.
            """
        )
    st.write("---")
    st.header("Project 4")
    st.write(
            """
            Fourth, I used the Keras functionality of Tensorflow to create a model that predicts the age of fossils based off of a number of features. This project was a fun way to learn about keras with a unique dataset from the kaggle platform. More can be found on the Keras Fossil Prediction tab.
            """
        )
    st.write("---")
    st.header("Project 5")
    st.write(
            """
            For the fifth project in the list, I used base TensorFlow to create a nerual network that can predict bankruptcy from a number of features. I leveraged sklearn to preprocess the data and select features for the model based off of importance rankings from the RandomForestClassifier for example. More information about this model and process can be found on the Tensorflow bankruptcy tab.
            """
        )
    st.write("---")
    st.header("Project 6")
    st.write(
            """
            Lastly, I used the Pytorch library to create a model that can classify clothing items. This project was a fun way to learn about the Pytorch library and how to create a CNN model that can be trained off of a large dataset. Additionally, I added a feature to the model that allows the user to upload their own image to classify. More information about this project can be found on the Pytorch Fashion MNIST tab. 
            """
        )
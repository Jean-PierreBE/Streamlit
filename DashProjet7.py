# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 20:20:15 2021

@author: stein
"""
import streamlit as st
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from PIL import Image

import json

import joblib
import shap

import sklearn
from sklearn.ensemble import GradientBoostingClassifier

PARAM = "Libels"
FILEMODEL = "joblib_Explainer.bz2"

ImgCredit = Image.open("ImgPret.jpg")
ImgSummary = Image.open("importance.JPG")

st.set_page_config(page_title="Prêt à dépenser", page_icon=ImgCredit,)

exload = joblib.load(filename=FILEMODEL)


def main():
    # chargement et préparation données
    datalib = load_libel()
    df_dash, Ikeys, df_result, df_predict, df_mean, shap_values_tst = load_data(
        datalib['File'], datalib['Col'])
    lst_var = load_list_var(df_dash)
    # Display header.
    st.markdown("<br>", unsafe_allow_html=True)

    # Affichage sidebar
    st.sidebar.image(ImgCredit, width=160)

    option = st.sidebar.selectbox("Client Nr", Ikeys)
    Lang = st.sidebar.selectbox("Language", datalib['Languages'])
    st.sidebar.image(ImgSummary, width=480)

    # Affichage haut de page
    st.title(datalib['TitlePg'][Lang] + str(option))

    i = Ikeys.loc[Ikeys.SK_ID_CURR == option].index
    prob = df_predict.loc[option]

    if df_result.loc[option] == 0:
        st.markdown(datalib['Accord'][Lang], unsafe_allow_html=True)
    else:
        st.markdown(datalib['Refus'][Lang], unsafe_allow_html=True)

    # Affichage interprétation modèle
    fig = shap.decision_plot(
        exload.expected_value, shap_values_tst[i, :], df_dash.loc[[option]], show=False)
    plt.savefig('scratch.JPG', format="JPG", bbox_inches='tight')

    ImgScratch = Image.open("scratch.JPG")
    st.image(ImgScratch, width=640,
             caption=datalib['LibGraf'][Lang] + str(prob))

    # Affichage
    Var = st.selectbox(datalib['SelVar'][Lang], lst_var)

    # Add histogram data
    # Setting title, labels, etc.
    ar = np.array([[df_dash.loc[option][Var]], [
                  df_mean.loc[Var][0]], [df_mean.loc[Var][1]]])
    df = pd.DataFrame(
        ar, index=[str(option), 'Target = 0', 'Target = 1'], columns=[Var])

    fig, ax = plt.subplots()
    ax.bar(df.index, df[Var])
    ax.set(title=datalib['LibComp'][Lang] + Var,
           ylabel=datalib['LibVal'][Lang], xlabel=datalib['LibXlabel'][Lang])
    ax.set_xticklabels(df.index, rotation=60)

    st.pyplot(fig)


@st.cache
def load_libel():
    f = open(PARAM, "r")
    # Reading from file
    dataj = json.loads(f.read())
    
    return dataj

@st.cache
def load_list_var(idf):
    list_col = list(idf)
    converted_list = [x.lower() for x in list_col]
    converted_list.sort()
    #list_col.sort()
    return converted_list

@st.cache
def load_data(iname, icol):
    df_file = pd.read_csv(iname["FILENAME"], delimiter=",", na_values=[
                          '-'], encoding="utf-8", low_memory=False)
    df_keys = df_file[[icol["Key"]]].copy()
    df_file.set_index(icol["Key"], inplace=True)
    df_file_result = df_file[icol["Target"]]
    df_file_predict = df_file[icol["Predict"]]
    df_file = df_file.drop(columns=[icol["Target"], icol["Predict"]])
    df_file.columns = map(str.lower, df_file.columns)
    # df_mean
    df_filem = pd.read_csv(iname["FILEMEAN"], delimiter=",", na_values=[
                           '-'], encoding="utf-8", low_memory=False)
    df_filem.set_index(icol["Var"], inplace=True)
    # shap values
    df_shap = pd.read_csv(iname["FILESHAP"], delimiter=",", na_values=[
                          '-'], encoding="utf-8", low_memory=False)

    return df_file, df_keys, df_file_result, df_file_predict, df_filem, df_shap.to_numpy()


if __name__ == '__main__':
    main()

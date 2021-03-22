# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 20:20:15 2021

@author: stein
"""
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import ToolGraph  as tg
#import numpy as np
import pandas as pd

from PIL import Image

import joblib
import shap


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

ImgCredit = Image.open("ImgPret.JPG")

FILENAME = "f_xtest_model.csv"
FILEMODEL = "joblib_Explainer.bz2"
FILESHAP = "f_shap_values.csv"

st.set_page_config(
    page_title="Scoring Prêt à dépenser", page_icon=ImgCredit,
)

exload = joblib.load(filename=FILEMODEL)

df_dash = pd.read_csv(FILENAME, delimiter=",", na_values=['-'], encoding = "utf-8", low_memory=False)

Ikeys = df_dash[["SK_ID_CURR"]].copy()

df_dash_stat = df_dash.copy()

df_dash.set_index("SK_ID_CURR" , inplace = True)
df_result = df_dash["Target"]
df_predict = df_dash["Predict"]
df_dash = df_dash.drop(columns = ["Target","Predict"])

df_shap = pd.read_csv(FILESHAP, delimiter=",", na_values=['-'], encoding = "utf-8", low_memory=False)

shap_values_tst = df_shap.to_numpy()

fig = shap.summary_plot(shap_values_tst, df_dash, plot_type="bar")
plt.savefig('summary.JPG',format="JPG", bbox_inches = 'tight')

# Display header.
st.markdown("<br>", unsafe_allow_html=True)
st.sidebar.image(ImgCredit, width=160)

st.title('Scoring pour accord de crédit')
st.markdown("Choisissez un client et voyez si le crédit peut-être accordé ou non.")

option = st.sidebar.selectbox('Choisissez un identifiant client',Ikeys)

i = Ikeys.loc[Ikeys.SK_ID_CURR == option].index

if df_result.loc[option] == 0:
    st.sidebar.markdown("statut du crédit : accordé")
else:
    st.sidebar.markdown("statut du crédit : refusé")
ImgSummary = Image.open("summary.JPG")
st.sidebar.image(ImgSummary, width=480)

#st.subheader("Graphe d'interprétation pour le client " + str(option) ) 

#st_shap(shap.force_plot(exload.expected_value, shap_values_tst[i,:], df_dash.loc[[option]]))

fig = shap.decision_plot(exload.expected_value, shap_values_tst[i,:], df_dash.loc[[option]],show=False)
plt.savefig('scratch.JPG',format="JPG", bbox_inches = 'tight')

ImgScratch = Image.open("scratch.JPG")
st.image(ImgScratch, width=640,caption = "Graphe d'interprétation pour le client " + str(option) )

colkeep = ["SK_ID_CURR","Target"]
fig= tg.calc_pie(df_dash_stat.copy(),colkeep,"nombre des crédits","Statut","count")
plt.savefig('pie.JPG',format="JPG", bbox_inches = 'tight')
ImgPie = Image.open("pie.JPG")
st.image(ImgPie, width=320)
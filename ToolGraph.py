# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:27:59 2020

@author: stein
"""
import matplotlib.pyplot as plt

import seaborn as sns
import gc
import pandas as pd

# graph to see the repartition of filled zones per period
def GraphPeriode(idf,ix,iy,ititle,icolor,isens):
    fig, ax=plt.subplots()
    ax.set_title(ititle, fontsize=45)
    ax.set_xlabel('',fontsize=25)
    ax.set_ylabel('',fontsize=30)
    fig.set_size_inches(30,8)
    sns.set_style("whitegrid")
    sns.set_context("paper")
    nb_indicateurs_par_annee=sns.barplot(x=ix, y=iy, data=idf,color=icolor,orient=isens)

def calc_pie(idf,icolkeep,ititle,inewlib,iaction):
    #tp.delColDf(idf,list(idf),icolkeep)
    idf1 = idf[icolkeep]
    idf1.drop_duplicates(inplace = True) 
    idf1.rename(columns={icolkeep[1]: inewlib,icolkeep[0]: ititle}, inplace=True)
    if iaction == "count":
        df_stat = idf1.groupby(inewlib).count()
    if iaction == "sum":
        df_stat = idf1.groupby(inewlib).sum()
    df_stat.plot.pie(y=ititle, figsize=(10, 10),autopct='%1.1f%%')
    plt.title("Répartition par " + ititle)
    gc.enable()
    del idf1,idf
    gc.collect()

def f_ShowPlot(idf,itarget,ivar,iylabel,ititle,ixlabel):
    #tp.delColDf(idf,list(idf),[ivar,itarget])
    idfi = idf[[ivar,itarget]]
    idf0 = idfi.loc[idf[itarget] == 0].groupby(ivar).count()
    idf0.rename(columns={itarget: itarget + " == 0"}, inplace=True)
    idf0.reset_index(inplace=True)
    
    idf1 = idfi.loc[idf[itarget] == 1].groupby(ivar).count()
    idf1.rename(columns={itarget: itarget + " == 1"}, inplace=True)
    idf1.reset_index(inplace=True)
    
    idf2 = pd.merge(idf0, idf1, on=ivar)
    idf2.set_index(ivar,inplace=True)
    
    idf2.plot(kind="barh")
    plt.title(ititle)
    plt.xlabel(ixlabel)
    plt.ylabel(iylabel)
    gc.enable()
    del idf,idfi
    gc.collect()
    
def graphic_variables(idf,icol,ititle,ixlabel):
    # Set the style of plots
    plt.style.use('fivethirtyeight')

    # Plot the distribution of idf
    plt.hist(idf[icol], edgecolor = 'k', bins = 25)
    plt.title(ititle); plt.xlabel(ixlabel); plt.ylabel('Nombre');
    
    plt.figure(figsize = (5, 4))
    plt.show()

    # KDE plot of loans that were repaid on time
    sns.kdeplot(idf.loc[idf['TARGET'] == 0, icol], label = 'target == 0')

    # KDE plot of loans which were not repaid on time
    sns.kdeplot(idf.loc[idf['TARGET'] == 1, icol], label = 'target == 1')

    # Labeling of plot
    plt.xlabel(ixlabel); plt.ylabel('Densité'); plt.title('Distribution par ' + ititle);
    plt.show()
    
"""
Cluster: ¿Dónde encaja Chile?
Agosto de 2021

Fuente originales:
-Texto: https://towardsdatascience.com/factor-analysis-cluster-analysis-on-countries-classification-1bdb3d8aa096
-Código y datos: https://github.com/ngaiyin1760/Factor-analysis-Cluster-analysis-on-countries-classification

Descripción datos usados:
-promedio en variables (no sólo último dato)
-excluye outliers (Argentina, Suiza, Hong Kong, Grecia, Turquia)
-division en pib per capita mayor/menor a 20k, A. Saudita y Rep. Checa en nivel bajo; 
"""


# ============================= Librerias =====================================


import pandas as pd
from pandas import read_csv
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
sns.set_style("darkgrid")
import os
import time


# import factor analyzer library
from factor_analyzer import FactorAnalyzer


# import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


#definir carpeta de trabajo en computador local
os.chdir('C:/Escritorio/Clusters')



# ========================  Armar Dataframe   =================================


# Variables con promedios, PIB per capita menor a 20,000
df_final = pd.read_excel('bdcluster_conpib.xlsx', sheet_name = 'menor20') 

# Índice para dataframe con nombre (necesario para correr matriz correlación)
df_final = df_final.set_index('Pais') 


#============================ Analisis de factores ============================


#Plot correlation matrix of indicators
plt.figure(figsize=(8,8))
corrMatrix = df_final.T.corr()
sns.heatmap(corrMatrix)


# Empieza el analisis de factores
fa = FactorAnalyzer()
fa.fit(df_final.T, 4)    # parto con cuatro factores


# calcula eigenvalores
ev, v = fa.get_eigenvalues()
print(ev)


# Create scree plot using matplotlib
plt.figure(figsize=(6,4))
plt.scatter(range(1,df_final.T.shape[1]+1),ev)
plt.plot(range(1,df_final.T.shape[1]+1),ev)
plt.hlines(1, 0, df_final.T.shape[1], colors='r')
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()


# Perform Factor Analysis
fa = FactorAnalyzer(list(ev >= 0.95).count(True), rotation='varimax')
fa.fit(df_final.T)
loads = fa.loadings_
loads = pd.DataFrame(loads, index=df_final.index)
print(loads)


# Guardar eigenvector en Excel (ie, loadings)
wrtr1 = pd.ExcelWriter('eigenvector.xlsx')
loads.to_excel(wrtr1, 'hoja1')
wrtr1.save()


#Heatmap of loadings
plt.figure(figsize=(8,8))
sns.heatmap(loads, annot=False, cmap="vlag")

# Get variance of each factor
fa_var = fa.get_factor_variance()
fa_var = pd.DataFrame(fa_var, index=['SS loadings', 'Proportion Var', 'Cumulative Var'])
print(fa_var)



#============================= Analisis de clusters=================================

#standardization along columns
df_final_std=(df_final.T-df_final.T.mean())/df_final.T.std()


#Create dendrogram
dendrogram = sch.dendrogram(sch.linkage(df_final_std, method='ward'))
plt.axhline(y=5.5, color='r', linestyle='--')


"""
A continuación definir numero clusters segun dendrograma: "n_clusters=5"
"""

# create clusters
hc = AgglomerativeClustering(n_clusters=5, affinity = 'euclidean', linkage = 'ward')
print(hc)

# save clusters for chart
y_hc = hc.fit_predict(df_final_std)


df_final_T = df_final.T
df_final_T['cluster'] = y_hc
df_final_T.sort_values("cluster", inplace = True, ascending=True)


# Para ver cómo van quedando los clusters (tabla preliminar)
# Al final el código genera un output con el archivo de excel de clusters
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(df_final_T['cluster'])


# Guardar resultados preliminares en Excel
nombrewriter = pd.ExcelWriter('resultadocluster.xlsx')
df_final_T.to_excel(nombrewriter, 'hoja1')
nombrewriter.save()


#agregar variable cluster al dataframe y ordenar
df_final_std['cluster'] = y_hc
df_final_std.sort_values("cluster", inplace = True, ascending=True)

df_cluster = df_final_T.groupby('cluster').mean()
df_cluster_std = df_final_std.groupby('cluster').mean()
print(df_cluster_std)

#Heatmap of cluster characteristics
plt.figure(figsize=(8,8))
# sns.heatmap(df_cluster_std.T, cmap="Blues", linewidths=.5)
sns.heatmap(df_cluster_std.T, cmap="vlag", linewidths=.5)



# =================  Ordenar dataframes antes de exportar  ====================


#count the number of countries in cluster
num_of_countries = []
for n in range(len(set(y_hc))):
    num_of_countries.append(sum(df_final_T['cluster'] == n))
    
df_cluster['num of countries'] = num_of_countries
df_cluster_std['num of countries'] = num_of_countries

columns = list(df_cluster.columns)
columns = columns[-1:] + columns[:-1]

df_cluster = df_cluster.reindex(columns=columns)
df_cluster_std = df_cluster_std.reindex(columns=columns)


# ======================== Exporta resultados a Excel  ========================


output = 'output_' + time.asctime(time.localtime(time.time())).replace(' ','_').replace(':','') + '.xlsx'

with pd.ExcelWriter(output) as writer:
    df_cluster.to_excel(writer, sheet_name='Cluster')
    df_cluster_std.to_excel(writer, sheet_name='Cluster_std')
    # df_cluster_std_selected.to_excel(writer, sheet_name='Cluster_std_selected')
    df_final_T.to_excel(writer, sheet_name='Result')
    df_final_std.to_excel(writer, sheet_name='Result_std')
    df_final.to_excel(writer, sheet_name='Final data')
    loads.to_excel(writer, sheet_name='Loading')
    # fa_transform.to_excel(writer, sheet_name='transf_factores')          
    # dfpd_pca.to_excel(writer, sheet_name='transf_pca')                   
    # df_pca_eigenvector.to_excel(writer, sheet_name='eigenvector_pca')    
    # corrMatrix.to_excel(writer, sheet_name='Correlation Matrix')
    #  df_dropped.to_excel(writer, sheet_name='Dropped')
    #  df.to_excel(writer, sheet_name='Raw data')
    #  datalog.to_excel(writer, sheet_name='Datalog')
    #  datalog_selected.to_excel(writer, sheet_name='Datalog_selected')


#================================  fin  =======================================
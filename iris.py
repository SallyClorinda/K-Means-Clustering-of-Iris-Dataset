#coding ini saya upload, hanya untuk belajar dan titip file coding.

#untuk import library pada python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

#untuk menampilkan semua data tabel dataset
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#untuk membaca file csv
iris = pd.read_csv("iris-data.csv", delimiter=';')
iris

iris.info()
iris[0:5]

#Frequency distribution of species"
iris_outcome = pd.crosstab(index=iris["class"],  # Make a crosstab
                              columns="count")      # Name the count column

iris_outcome

iris_setosa=iris.loc[iris["class"]=="Iris-setosa"]
iris_virginica=iris.loc[iris["class"]=="Iris-virginica"]
iris_versicolor=iris.loc[iris["class"]=="Iris-versicolor"]

sns.FacetGrid(iris,hue="class",size=3).map(sns.distplot,"petal_length").add_legend()
sns.FacetGrid(iris,hue="class",size=3).map(sns.distplot,"petal_width").add_legend()
sns.FacetGrid(iris,hue="class",size=3).map(sns.distplot,"sepal_length").add_legend()
plt.show()

sns.set_style("whitegrid")
sns.pairplot(iris,hue="class",size=3);
plt.show()


#sumber: https://www.kaggle.com/code/khotijahs1/k-means-clustering-of-iris-dataset 

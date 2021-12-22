




# IMPORT LIBRARIES


import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from numpy.random import randn
from numpy.random import seed
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd 
import os
import math
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
#from sklearn.externals import joblib
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import warnings; warnings.simplefilter('ignore')
#matplotlib inline

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



#READING THE DATASET

data=pd.read_excel("C:/Users/Student/Desktop/ML/data.xlsx")

data1=pd.read_csv("C:/Users/Student/Desktop/ML/data1.csv",names=['userId','productId','Rating','likes_count'])
data1.head(5) 


data1.Rating.min()
data1.Rating.max()
data1.info()
data1.shape


#CHECK FOR NULL VALUES

print('Number of missing values across columns: \n',data1.isnull().sum())


#Popularity Based Recommendation

#Getting the new dataframe which contains users who has given 50 or more ratings


data2=data1.groupby("productId").filter(lambda x:x['Rating'].count() >=50)

no_of_ratings_per_product = data2.groupby(by='productId')['Rating'].count().sort_values(ascending=False)
no_of_ratings_per_product

fig = plt.figure(figsize=plt.figaspect(.5))
ax = plt.gca()
plt.plot(no_of_ratings_per_product.values)
plt.title('# RATINGS per Product')
plt.xlabel('Product')
plt.ylabel('No of ratings per product')
ax.set_xticklabels([])

plt.show()

#Average rating of the product 
   
data2.groupby('productId')['Rating'].mean().head()
data2.groupby('productId')['Rating'].mean().sort_values(ascending=False).head()


#Total no of rating for product
data2.groupby('productId')['Rating'].count().sort_values(ascending=False).head()
ratings_mean_count = pd.DataFrame(data2.groupby('productId')['Rating'].mean())
ratings_mean_count['rating_counts'] = pd.DataFrame(data2.groupby('productId')['Rating'].count())
ratings_mean_count.head()
ratings_mean_count['rating_counts'].max()

plt.figure(figsize=(8,6))
plt.xlabel('Rating Count')
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['rating_counts'].hist(bins=50)

plt.figure(figsize=(8,6))
plt.xlabel('Rating')
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['Rating'].hist(bins=50)

plt.figure(figsize=(8,6))
plt.xlabel('Rating')
plt.ylabel('Rating Count')
plt.rcParams['patch.force_edgecolor'] = True
sns.jointplot(x='Rating', y='rating_counts', data=ratings_mean_count, alpha=0.4)


popular_products = pd.DataFrame(data2.groupby('productId')['Rating'].count())
most_popular = popular_products.sort_values('Rating', ascending=False)
most_popular.head(30).plot(kind = "bar")


# COLLOBORATIVE FILTERING

with sns.axes_style('white'):
    g = sns.factorplot("Rating", data=data1, aspect=2.0,kind='count')
    g.set_ylabels("Total number of ratings")
#Most of the people has given the rating of 5


print("\nTotal no of ratings :",data1.shape[0])
print("Total No of Users   :", len(np.unique(data1.userId)))
print("Total No of products  :", len(np.unique(data1.productId)))

#Analysis of rating given by the user 
no_of_rated_products_per_user = data1.groupby(by='userId')['Rating'].count().sort_values(ascending=False)
no_of_rated_products_per_user.head(10)

no_of_rated_products_per_user.describe()

quantiles = no_of_rated_products_per_user.quantile(np.arange(0,1.01,0.01), interpolation='higher')

plt.figure(figsize=(10,10))
plt.xlabel('Quantile')
quantiles.plot()


# quantiles with 0.05 difference
plt.scatter(x=quantiles.index[::5], y=quantiles.values[::5], c='orange', label="quantiles with 0.05 intervals")

# quantiles with 0.25 difference
plt.scatter(x=quantiles.index[::25], y=quantiles.values[::25], c='m', label = "quantiles with 0.25 intervals")
plt.ylabel('No of ratings by user')
plt.xlabel('Value at the quantile')
plt.legend(loc='best')
plt.show()

print('\n No of rated product more than 50 per user : {}\n'.format(sum(no_of_rated_products_per_user >= 50)) )

#ACCURACY


#Model-based collaborative filtering system

data3=data2.head(10000)
X = data3.pivot_table(values='Rating', index='userId', columns='productId', fill_value=0)
X.head()

X.shape

X1 = X.T
X.head()

X1.shape



# Unique products in subset of data

X1=X

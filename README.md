# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE

Developed By: kanimozhi.s
Reg.No: 212221220024
~~~
# loading dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("titanic_dataset.csv")
df

#checking data
df.isnull().sum()


#removing unnecessary data variables
df.drop('Cabin',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Parch',axis=1,inplace=True)
#cleaning data
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()

#removing outliers 
plt.title("Dataset with outliers")
df.boxplot()
plt.show()

cols = ['Age','SibSp','Fare']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()

#feature encoding 
from sklearn.preprocessing import OrdinalEncoder
embark=["C","S","Q"]
emb=OrdinalEncoder(categories=[embark])
df["Embarked"]=emb.fit_transform(df[["Embarked"]])

from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
df


#feature scaling
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df2=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])
df2

#feature transformation
df2.skew()


import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)
#no skew data that need no transformation- Age, Embarked
#moderately positive skew- Survived, Sex
#highy positive skew- Fare, SibSp
#highy negative skew- Pclass
df1=pd.DataFrame()
df1["Survived"]=np.sqrt(df2["Survived"])
df1["Pclass"],parameters=stats.yeojohnson(df2["Pclass"])
df1["Sex"]=np.sqrt(df2["Sex"])
df1["Age"]=df2["Age"]
df1["SibSp"],parameters=stats.yeojohnson(df2["SibSp"])
df1["Fare"],parameters=stats.yeojohnson(df2["Fare"])
df1["Embarked"]=df2["Embarked"]
df1.skew()

#feature selection process
import matplotlib
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1) 
y = df1["Survived"]          

plt.figure(figsize=(12,10))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)
plt.show()

cor_target = abs(cor["Survived"])
relevant_features = cor_target[cor_target>0.5]
relevant_features

X_1 = sm.add_constant(X)
model = sm.OLS(y,X_1).fit()
model.pvalues


#Backward Elimination
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)

model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 4)
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)

nof_list=np.arange(1,6)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))

cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model, 2)             
X_rfe = rfe.fit_transform(X,y)  
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)

#Embedded Method
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()

~~~

# OUPUT

Data Preprocessing before Feature Selection:

# Initial Dataset:

![image](https://user-images.githubusercontent.com/129577149/236849353-f3b3e816-4aab-4436-be94-28afaab2bf99.png)

# Data checking and cleaning:

![image](https://user-images.githubusercontent.com/129577149/236849480-5aeedb0c-16ad-41da-a472-b38928eefb5e.png)

![image](https://user-images.githubusercontent.com/129577149/236849546-8590ddc7-7aac-4fd9-a440-7064310dcbe6.png)

# Outlier Removal

![image](https://user-images.githubusercontent.com/129577149/236849628-cca71dc8-5e61-44b5-9b71-9f2777243703.png)

![image](https://user-images.githubusercontent.com/129577149/236849726-b73f5f7c-ce6e-4512-a66f-442be81743ab.png)

# Feature Enoding:

![image](https://user-images.githubusercontent.com/129577149/236849828-e1b493ce-85db-49ac-82b7-4e447eb47e7a.png)

# Feature Transformations:

![image](https://user-images.githubusercontent.com/129577149/236849971-7e1f9abb-b6e3-441d-8d60-64637f83effe.png)

![image](https://user-images.githubusercontent.com/129577149/236850120-fa6eed4a-745f-49ab-b9a6-f6c2ebfceab1.png)

# Feature Selection

# Filter Method

The filtering here is done using correlation matrix and it is most commonly done using Pearson correlation.

![image](https://user-images.githubusercontent.com/129577149/236850739-8dd18219-ca59-4324-a9d2-cbfa09096580.png)


# Highly correlated features with the Output variable

![image](https://user-images.githubusercontent.com/129577149/236850797-30379471-189f-4b37-a807-83cb50f60a1d.png)

# Wrapper Method:

Wrapper Method is an iterative and computationally expensive process but it is more accurate than the filter method.

There are different wrapper methods such as Backward Elimination, Forward Selection, Bidirectional Elimination and RFE.

# Backward Elimination:

![image](https://user-images.githubusercontent.com/129577149/236851042-af8207f3-71a5-4a92-ae2b-0e28d8a75959.png)

![image](https://user-images.githubusercontent.com/129577149/236851104-3e81acba-e120-4f99-b9e8-4d89460020c8.png)

# RFE (Recursive Feature Elimination):

![image](https://user-images.githubusercontent.com/129577149/236851231-5346e2b8-7cc1-40f0-af50-ffc797b351e7.png)

# Optimum number of features that have high accuracy:

![image](https://user-images.githubusercontent.com/129577149/236851339-fb1cf47b-ab4d-4d5f-b8f8-954c7141309c.png)

# Final set of feature:

![image](https://user-images.githubusercontent.com/129577149/236851447-38bf9812-62ca-473c-84df-ba9dc0fa21f5.png)

# Embedded Method:

Embedded methods are iterative in a sense that takes care of each iteration of the model training process and carefully extract those features which contribute the most to the training for a particular iteration. Regularization methods are the most commonly used embedded methods which penalize a feature given a coefficient threshold.

![image](https://user-images.githubusercontent.com/129577149/236851723-eb9c87d3-15df-43ab-8a86-9e14701e79e9.png)

# RESULT:

Thus, the various feature selection techniques have been performed on a given dataset successfully.







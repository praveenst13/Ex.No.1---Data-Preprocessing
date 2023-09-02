# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
```Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train
```

## PROGRAM:

```
REG NO:212222240077
NAME:Praveen s

#import necessary libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#read the datset
df=pd.read_csv("/content/Churn_Modelling.csv")
print(df)

#dropout unwanted columns
df1=df.drop(['RowNumber','Age','Geography','Surname','Gender'],axis=1)
df1

#checking for null values
print(df1.isnull().sum())
df1.fillna(df.mean().round(1),inplace=True)
print(df1.duplicated())

#normalize the data
scalar=MinMaxScaler()
df2=pd.DataFrame(scalar.fit_transform(df1))
df2

#split the datset as x and y
x=df2.iloc[:,:-1].values
print(x)
y=df2.iloc[:,-1].values
print(y)

#split the dataset for training and testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_test)
print(x_train)
print(len(x_train))
print(len(x_test))

```

## OUTPUT:
## dataset
![image](https://github.com/Ragu-123/Ex.No.1---Data-Preprocessing/assets/113915622/aac4a645-0b09-475a-a5c6-62f9b2b791bf)
## checking for null values
![image](https://github.com/Ragu-123/Ex.No.1---Data-Preprocessing/assets/113915622/9f2f5ff9-41a6-49cc-97af-21b508d6d00f)
## normalize the data
![image](https://github.com/Ragu-123/Ex.No.1---Data-Preprocessing/assets/113915622/b861d35c-69b3-455d-ad00-e06dd064692f)
## split the datset as x and y
![image](https://github.com/Ragu-123/Ex.No.1---Data-Preprocessing/assets/113915622/2803422c-0f3e-487a-ba7e-8e263e1e0adc)
## split the dataset for training and testing
![image](https://github.com/Ragu-123/Ex.No.1---Data-Preprocessing/assets/113915622/69849559-9f53-47a1-a895-0968de260afb)

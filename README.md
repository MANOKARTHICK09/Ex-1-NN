<H3>MANO KARTHICK S</H3>
<H3>212222230077</H3>
<H3>EX. NO.1</H3>
<H3>3/7/2024</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df
df.isnull().sum()
df.duplicated()
df.describe()
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```


## OUTPUT:
![image](https://github.com/MANOKARTHICK09/Ex-1-NN/assets/121785458/37740be0-236c-453a-a37b-b4ac5106c6d7)
![image](https://github.com/MANOKARTHICK09/Ex-1-NN/assets/121785458/2a1239a6-48a8-4754-ba18-6b1ba6e6a84c)
![image](https://github.com/MANOKARTHICK09/Ex-1-NN/assets/121785458/f5b20fde-9196-49a6-866e-8288e92e1e5f)
![image](https://github.com/MANOKARTHICK09/Ex-1-NN/assets/121785458/5dfb5e17-5fa4-4df3-934b-17ade8f2dd9a)
![image](https://github.com/MANOKARTHICK09/Ex-1-NN/assets/121785458/39f6f42a-02e1-4753-8ab2-58ecabf25ed5)
![image](https://github.com/MANOKARTHICK09/Ex-1-NN/assets/121785458/80281a45-4c14-4b8f-8be7-1726fa6ae5f8)
![image](https://github.com/MANOKARTHICK09/Ex-1-NN/assets/121785458/0ccbdea4-53ef-43ac-aec1-45f66c53202c)
![image](https://github.com/MANOKARTHICK09/Ex-1-NN/assets/121785458/3e118b80-5570-4a64-bcf6-cdab48a18014)
![image](https://github.com/MANOKARTHICK09/Ex-1-NN/assets/121785458/b60848f4-e966-4381-83e7-dd6d99a1a957)



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.



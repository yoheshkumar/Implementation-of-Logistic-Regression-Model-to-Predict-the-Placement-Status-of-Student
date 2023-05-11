# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the data and use label encoder to change all the values to numeric.
2. Classify the training data and the test data.
3. Calculate the accuracy score, confusion matrix and classification report.
4. Then program predicts the Logistic Regression.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: YOHESH KUMAR R.m
RegisterNumber:  212222240118
*/
```
```
*/
import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1) #removes the specified row or cols
data1.head() 

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear") 
lr.fit(x_train,y_train)
y_pred= lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1= classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
### Placement Data :
![p d1](https://github.com/yoheshkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393568/689d7b96-9edb-4d54-9b50-963a52a19bb6)
### Salary Data :
![sd 1](https://github.com/yoheshkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393568/36c24e07-f234-43a9-ab07-604db82f9eb2)
### isnull() :
![isnul](https://github.com/yoheshkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393568/3ae968b1-d490-4bcf-a7d0-597f97029c42)
### Checking For Duplicates :
![cfd](https://github.com/yoheshkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393568/de8142c7-c5a7-4999-97fd-15e9f21dd037)
### Print Data :
![pd](https://github.com/yoheshkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393568/a4c267a1-c194-4d74-9692-3653171b6ee9)
### Data Status :
![ds](https://github.com/yoheshkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393568/0ab6a6ea-8941-41f5-822d-3a1d50d686d7)
### y_prediction Array :
![y](https://github.com/yoheshkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393568/624eeb35-1fd7-4fb1-9a3f-4cf880a33a6b)
### Accuracy Score :
![as](https://github.com/yoheshkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393568/b8009461-93bb-4372-b668-79ab7991c063)
### Confusion Matrix :
![cm](https://github.com/yoheshkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393568/5ff03182-81c0-4d65-a107-ab1397fd18c2)
### Classification Report :
![cr](https://github.com/yoheshkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393568/a35cb629-711e-4e5b-b901-2d03cbc33e31)
### Prediction of LR :
![plr](https://github.com/yoheshkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393568/e3930df1-9bca-4796-9eef-294c1cb8ef14)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

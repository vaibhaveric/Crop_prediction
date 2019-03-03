#!/usr/bin/env python
# coding: utf-8

# In[104]:

# Importing Necessary Libraries
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler


# In[114]:

def model():
    df1 = pd.read_csv('data.csv') #Reading CSV File
    df1.fillna(df1.mean(),inplace=True)
    col = df1.columns.values #Getting Column Names
    dates = col[2:-1]
    new_dates = []
    # Spliting the dates
    for i in dates:
        a = i.split('-')
        new_dates.append(float("".join(a)))

    wght = df1['Weight'].tolist() # Converting DataFrame into List
    com = df1['COMMODITIES'].tolist()
    data  = []
    for i in dates:
        data.append(df1[i].tolist())
    voc = dict((c,i+1)for i,c in enumerate(com)) #Assigning the Feature a unique value
    new_com = [voc[i] for i in com]
    new_data = zip(*data)
    con_data = [list(i) for i in new_data]
    final_target = []
    feature = [list(i) for i in zip(new_com,wght)]
    new_few = []
    for i in feature:
        for j in new_dates:
            s=i
            new_few.append(s+[j])
    for i in con_data:
        for j in i:
            final_target.append(j)
    sc = StandardScaler() # Scalering the Feature of data
    X = sc.fit_transform(new_few)
    x_train,x_test,y_train,y_test = train_test_split(X,final_target) #Spliting The train test split
    print("Trainning Started")
    mlp = MLPRegressor() # Creating DecisionTreeRegressor Instances
    param = {'activation' : ['logistic','tanh','relu'],'solver' : ['lbfgs', 'sgd', 'adam'],'alpha':[0.0001,0.1]} # This are the parameter for MLPRegressor
    grd = GridSearchCV(mlp,param_grid=param) # The GridSearch will try every parameter it will select which will give the best result 
    model = grd.fit(x_train,y_train) # Training the Model
    sc = grd.score(x_test,y_test) # Testing the Score and Getting Accuracy
    return model,voc,sc



# In[ ]:





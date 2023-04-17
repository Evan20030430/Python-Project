import pandas
import numpy as np
from sklearn import linear_model 
from collections import Counter
from sklearn.model_selection import train_test_split

''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path2.
'''
dataset_2 = pandas.read_csv('NYC_Bicycle_Counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Total']  = pandas.to_numeric(dataset_2['Total'].replace(',','', regex=True))
# print(dataset_2.to_string()) #This line will print out your data

# Problem 1: Install 3 Sensors
def question1():
    print("QUESTION 1\n************************")
    y = dataset_2['Total']
    x_Brook = []
    x_Man = []
    x_Queen = []
    x_Will = []
    x_Brook.append(dataset_2['Brooklyn Bridge'])
    x_Man.append(dataset_2['Manhattan Bridge'])
    x_Queen.append(dataset_2['Queensboro Bridge'])
    x_Will.append(dataset_2['Williamsburg Bridge'])

    # Collect and reshape all data
    x_Brook = np.array(x_Brook)
    x_Brook = x_Brook.reshape((-1, ))
    x_Man = np.array(x_Man)
    x_Man = x_Man.reshape((-1, ))
    x_Queen = np.array(x_Queen)
    x_Queen = x_Queen.reshape((-1, ))
    x_Will = np.array(x_Will)
    x_Will = x_Will.reshape((-1, ))
    y = np.array(y)

    # Approach 1: Find the max at each day, then use the top three bridges
    # that has the max the most
    values = {"Brook": 0, "Man": 0, "Queen": 0, "Will": 0}
    for i in range(len(y)):
        vals = [x_Brook[i], x_Man[i], x_Queen[i], x_Will[i]]
        idx = (vals.index(max(vals)))
        if idx == 0:
            values["Brook"] += 1
        elif idx == 1:
            values["Man"] += 1
        elif idx == 2:
            values["Queen"] += 1
        else:
            values["Will"] += 1

    k1 = Counter(values)
    max_3 = k1.most_common(3)
   
    print("Approach 1")
    for i in max_3:
        print("{}: {}" .format(i[0], i[1]))


    # Approach 2: Find each bridge percentage accross all data    
    Brook_sum = np.sum(x_Brook) 
    Man_sum = np.sum(x_Man)
    Queen_sum = np.sum(x_Queen)
    Will_sum = np.sum(x_Will)
    total = np.sum(y)
    bri_per = dict() # Percentage for each bridge
    bri_per["Brook"] = Brook_sum/total
    bri_per["Man"] = Man_sum/total
    bri_per["Queen"] = Queen_sum/total
    bri_per["Will"] = Will_sum/total
    
    k2 = Counter(bri_per)
    max_3_per = k2.most_common(3)
    
    print("\nApproach 2")
    for i in max_3_per:
        print("{} Percentage: {:.4f}" .format(i[0], i[1]))
    
    print("************************")

# Problem 2: Num Bicyclist Prediction
def question2():
    print("\n\nQUESTION 2\n************************")
    print("Approach 1")
    y = dataset_2['Total']
    x_high_temp = []
    x_low_temp = []
    x_precip = []
    x_high_temp.append(dataset_2['High Temp'])
    x_low_temp.append(dataset_2['Low Temp'])
    x_precip.append(dataset_2['Precipitation'])

    # Collect all feature data
    x_high_temp = np.array(x_high_temp)
    x_low_temp = np.array(x_low_temp)
    x_precip = np.array(x_precip)
    
    # Concatenate all data and reshape it
    X = np.vstack((x_high_temp, x_low_temp, x_precip))
    X = X.T
 
    # Train model and collect coefficients
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(X,y)
    print("Coefficients: High Temp: {:.4f}, Low Temp: {:.4f}, Precipitation: {:.4f}" .format(regr.coef_[0], regr.coef_[1], regr.coef_[2]))
    print("Intercept: {:.4f}" .format(regr.intercept_))
    
    # Calculate r2
    y_pred = []
    for i in range(len(x_high_temp)):
        y_pred.append(regr.coef_[0] * x_high_temp[i] + regr.coef_[1] * x_low_temp[i] + regr.coef_[2] * x_precip + regr.intercept_)
    
    y_pred = np.array(y_pred[0][0])
    r2 = 1 - SSR(y, y_pred) / SST(y)
    print("r2: {:.4f}" .format(r2))

    
    print("\n\nApproach 2")
    r2 = 1
    coeff = []
    intercept = []

    for i in range(100):
        y_pred = []
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True
        )
        regr = linear_model.LinearRegression(fit_intercept=True)
        regr.fit(X_train,y_train)
        for j in range(len(X_test)):
            y_pred.append(regr.coef_[0] * X_test[j][0] + regr.coef_[1] * X_test[j][1] + regr.coef_[2] * X_test[j][2] + regr.intercept_)
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        r2_test = 1 - SSR(y_test, y_pred) / SST(y_test)
        if (r2_test < r2 and r2_test > 0):
            r2 = r2_test
            coeff = regr.coef_
            intercept = regr.intercept_
    
    print("Coefficients: High Temp: {:.4f}, Low Temp: {:.4f}, Precipitation: {:.4f}" .format(coeff[0], coeff[1], coeff[2]))
    print("Intercept: {:.4f}" .format(intercept))
    print("r2: {:.4f}" .format(r2))

    
    print("************************")

    
    
def SSR(y_data, y_pred):
    # Calculate SSR with target data and prediction values
    SSR = 0
    for i in range(len(y_data)):
        SSR += np.square(y_data[i] - y_pred[i])
    
    return SSR

def SST(y_data):
    SST = 0
    y_bar = np.average(y_data)
    for i in range(len(y_data)):
        SST += np.square(y_data[i] - y_bar)
        
    return SST
    
question1()
question2()
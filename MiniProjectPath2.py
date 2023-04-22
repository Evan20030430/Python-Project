import pandas
import numpy as np
from sklearn import linear_model 
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

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
    
    # Approach 3: Train model with only 3 bridges, use r2 to find the best combination
    print("\nApproach 3")
    sum1 = x_Brook + x_Man + x_Queen # Brook, Man, Queen
    sum2 = x_Brook + x_Man + x_Will # Brook, Man, Will
    sum3 = x_Brook + x_Queen + x_Will # Brook, Queen, Will
    sum4 = x_Man + x_Queen + x_Will # Man, Queen, Will

    sum1 = sum1.reshape(-1, 1)
    sum2 = sum2.reshape(-1, 1)
    sum3 = sum3.reshape(-1, 1)
    sum4 = sum4.reshape(-1, 1)

    sum = [sum1, sum2, sum3, sum4]
    total = x_Brook + x_Man + x_Queen + x_Will
    r2 = 1
    index = 0

    for i in range(4):
        temp_coe, temp_int = train_model(sum[i], y)
        temp_r2 = cal_r2(y, total, temp_coe, temp_int)
        if temp_r2 < r2 and temp_r2 > 0:
            r2, index = temp_r2, i
    
    if (index == 0):
        print("Three optimal bridges: Brook, Man, Queen")
    elif (index == 1):
        print("Three optimal bridges: Brook, Man, Will")
    elif (index == 2):
        print("Three optimal bridges: Brook, Queen, Will")
    else:
        print("Three optimal bridges: Man, Queen, Will")

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
    
# Problem 3: Classify what day is today based on number of bicyclists
def question3():
    print("\n\nQUESTION 3\n************************")
    total_bicyclists = np.array(dataset_2["Total"]).reshape(-1,1)
    actual_days = np.array(dataset_2["Day"])
    X_train, X_test, y_train, y_test = train_test_split(total_bicyclists, actual_days, test_size=0.3, shuffle=False)
    
    # Approach 1: K-Nearest Neighbors With Different K
    print()
    print("Approach 1: K-Nearest Neighbors With Different K")
    print("--------------------------------------------------")

    for k in range(1,31):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        auc_scor = metrics.roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
        print()
        print(str(k) + "-Nearest Neighbors Results:")
        print(f"Accuracy Score: {acc}")
        print(f"AUROC Score: {auc_scor}")

    # Approach 2: Support Vector Machine
    print()
    print("Approach 2: Support Vector Machine")
    print("--------------------------------------------------")
    model = SVC(probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    auc_scor = metrics.roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    print()
    print(f"Accuracy Score: {acc}")
    print(f"AUROC Score: {auc_scor}")

    # Approach 3: Multilayer Perceptron
    print()
    print("Approach 3: Multilayer Perceptron")
    print("--------------------------------------------------")
    model = MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=(20,20))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    auc_scor = metrics.roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    print()
    print(f"Accuracy Score: {acc}")
    print(f"AUROC Score: {auc_scor}")

    # None of the approaches yield good results: attempting to plot the distribution of the data

    # Create a list of datapoints for each day to plot a histogram for each day.
    monlist = []
    tuelist = []
    wedlist = []
    thurlist = []
    frilist = []
    satlist = []
    sunlist = []

    for day, numcyclist in zip(np.array(dataset_2['Day']), np.array(dataset_2['Total'])):
        if day == 'Monday':
            monlist.append(numcyclist)
        elif day == 'Tuesday':
            tuelist.append(numcyclist)
        elif day == 'Wednesday':
            wedlist.append(numcyclist)
        elif day == 'Thursday':
            thurlist.append(numcyclist)
        elif day == 'Friday':
            frilist.append(numcyclist)
        elif day == 'Saturday':
            satlist.append(numcyclist)
        else:
            sunlist.append(numcyclist)

    # We will now plot a histogram for each day
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(30, 15))
    ax[0,0].hist(monlist, bins = int(np.sqrt(len(monlist))))
    ax[0,0].set_title("Monday")
    ax[0,0].set_xlabel("Number of Bicyclists")
    ax[0,0].set_ylabel("Frequency")


    ax[0,1].hist(tuelist, bins = int(np.sqrt(len(tuelist))))
    ax[0,1].set_title("Tuesday")
    ax[0,1].set_xlabel("Number of Bicyclists")
    ax[0,1].set_ylabel("Frequency")

    ax[0,2].hist(wedlist, bins = int(np.sqrt(len(wedlist))))
    ax[0,2].set_title("Wednesday")
    ax[0,2].set_xlabel("Number of Bicyclists")
    ax[0,2].set_ylabel("Frequency")

    ax[0,3].hist(thurlist, bins = int(np.sqrt(len(thurlist))))
    ax[0,3].set_title("Thursday")
    ax[0,3].set_xlabel("Number of Bicyclists")
    ax[0,3].set_ylabel("Frequency")

    ax[1,0].hist(frilist, bins = int(np.sqrt(len(frilist))))
    ax[1,0].set_title("Friday")
    ax[1,0].set_xlabel("Number of Bicyclists")
    ax[1,0].set_ylabel("Frequency")

    ax[1,1].hist(satlist, bins = int(np.sqrt(len(satlist))))
    ax[1,1].set_title("Saturday")
    ax[1,1].set_xlabel("Number of Bicyclists")
    ax[1,1].set_ylabel("Frequency")

    ax[1,2].hist(sunlist, bins = int(np.sqrt(len(sunlist))))
    ax[1,2].set_title("Sunday")
    ax[1,2].set_xlabel("Number of Bicyclists")
    ax[1,2].set_ylabel("Frequency")

    fig.delaxes(ax[1,3]) # Delete unwanted subplot
    fig.suptitle("Histograms Representing the Number of Bicyclists In Each Weekday")
    plt.show()

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

def train_model(x, y):
    # Return r2, coefficients, and intercept
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x,y)
    return regr.coef_[0], regr.intercept_

def cal_r2(y, test_data, coeff, intc):
    y_pred = []
    for i in range(len(test_data)):
        y_pred.append(coeff * test_data[i] + intc)
    y_pred = np.array(y_pred)
    r2 = 1 - SSR(y, y_pred) / SST(y)
    return r2

question1()
question2()
question3()
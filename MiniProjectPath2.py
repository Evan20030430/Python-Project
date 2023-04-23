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
from sklearn.preprocessing import StandardScaler


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
    # Train the data with three bridges (use different combinations) for Ridge regression
    lmbda = np.logspace(start=-1, stop=3, num=51, base=10.0)

    # Brooklyn, Manhattan, Williamsburg
    brook_arr = np.array(dataset_2["Brooklyn Bridge"])
    manhattan_arr = np.array(dataset_2["Manhattan Bridge"])
    will_arr = np.array(dataset_2["Williamsburg Bridge"])

    totals1 = []
    truetots = np.array(dataset_2["Total"])

    for brook, man, will in zip(brook_arr, manhattan_arr, will_arr):
        totals1.append(brook + man + will)

    X_train, X_test, y_train, y_test = train_test_split(
            totals1, truetots, test_size=0.2, shuffle=True
        )
    
    minMSE = 10e1000
    finr2 = None
    minMSEpred = None

    for l in lmbda:
        model = linear_model.Ridge(alpha=l, fit_intercept=True)
        model.fit(np.array(X_train).reshape(-1,1), y_train)
        y_pred = model.predict(np.array(X_test).reshape(-1,1))
        r2score = metrics.r2_score(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)

        if(mse < minMSE):
            minMSE = mse
            minMSEpred = y_pred
            finr2 = r2score

    print("\nBROOKLYN, MANHATTAN, WILLIAMSBURG")

    print(f"The r2 value for Brooklyn, Manhattan, Williamsburg: {finr2}")
    print(f"The MSE value for Brooklyn, Manhattan, Williamsburg: {minMSE}")

    plt.scatter(totals1, truetots)
    plt.plot(X_test, minMSEpred, 'k')
    plt.grid()
    plt.xlabel("Three bridge total")
    plt.ylabel("Four bridge total")
    plt.suptitle("Brooklyn, Manhattan,  Williamsburg")
    plt.show()

    # Manhattan, Williamsburg, Queensboro
    queen_arr = np.array(dataset_2["Queensboro Bridge"])

    totals2 = []

    for man, will, queen in zip(manhattan_arr, will_arr, queen_arr):
        totals2.append(man + will + queen)
    
    X_train, X_test, y_train, y_test = train_test_split(
            totals2, truetots, test_size=0.2, shuffle=True
        )
    
    minMSE = 10e1000
    finr2 = None
    minMSEpred = None

    for l in lmbda:
        model = linear_model.Ridge(alpha=l, fit_intercept=True)
        model.fit(np.array(X_train).reshape(-1,1), y_train)
        y_pred = model.predict(np.array(X_test).reshape(-1,1))
        r2score = metrics.r2_score(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)

        if(mse < minMSE):
            minMSE = mse
            minMSEpred = y_pred
            finr2 = r2score

    print("\nMANHATTAN, WILLIAMSBURG, QUEENSBORO")

    print(f"The r2 value for Manhattan, Williamsburg, Queensboro: {finr2}")
    print(f"The MSE value for Manhattan, Williamsburg, Queensboro: {minMSE}")

    plt.scatter(totals2, truetots)
    plt.plot(X_test, minMSEpred, 'k')
    plt.grid()
    plt.xlabel("Three bridge total")
    plt.ylabel("Four bridge total")
    plt.suptitle("Manhattan,  Williamsburg, Queensboro")
    plt.show()

    # Brooklyn, Williamsburg, Queensboro
    totals3 = []

    for brook, will, queen in zip(brook_arr, will_arr, queen_arr):
        totals3.append(brook + will + queen)
    
    X_train, X_test, y_train, y_test = train_test_split(
            totals3, truetots, test_size=0.2, shuffle=True
        )
    
    minMSE = 10e1000
    finr2 = None
    minMSEpred = None

    for l in lmbda:
        model = linear_model.Ridge(alpha=l, fit_intercept=True)
        model.fit(np.array(X_train).reshape(-1,1), y_train)
        y_pred = model.predict(np.array(X_test).reshape(-1,1))
        r2score = metrics.r2_score(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)

        if(mse < minMSE):
            minMSE = mse
            minMSEpred = y_pred
            finr2 = r2score

    print("\nBROOKLYN, WILLIAMSBURG, QUEENSBORO")

    print(f"The r2 value for Brooklyn, Williamsburg, Queensboro: {finr2}")
    print(f"The MSE value for Brooklyn, Williamsburg, Queensboro: {minMSE}")

    plt.scatter(totals3, truetots)
    plt.plot(X_test, minMSEpred, 'k')
    plt.grid()
    plt.xlabel("Three bridge total")
    plt.ylabel("Four bridge total")
    plt.suptitle("Brooklyn,  Williamsburg, Queensboro")
    plt.show()

    # Brooklyn, Manhattan, Queensboro
    totals4 = []

    for brook, man, queen in zip(brook_arr, manhattan_arr, queen_arr):
        totals4.append(brook + man + queen)
    
    X_train, X_test, y_train, y_test = train_test_split(
            totals4, truetots, test_size=0.2, shuffle=True
        )
    
    minMSE = 10e1000
    finr2 = None
    minMSEpred = None

    for l in lmbda:
        model = linear_model.Ridge(alpha=l, fit_intercept=True)
        model.fit(np.array(X_train).reshape(-1,1), y_train)
        y_pred = model.predict(np.array(X_test).reshape(-1,1))
        r2score = metrics.r2_score(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)

        if(mse < minMSE):
            minMSE = mse
            lowestMSEModel = model
            minMSEpred = y_pred
            finr2 = r2score

    print("\nBROOKLYN, MANHATTAN, QUEENSBORO")

    print(f"The r2 value for Brooklyn, Manhattan, Queensboro: {finr2}")
    print(f"The MSE value for Brooklyn, Manhattan, Queensboro: {minMSE}")

    plt.scatter(totals4, truetots)
    plt.plot(X_test, minMSEpred, 'k')
    plt.grid()
    plt.xlabel("Three bridge total")
    plt.ylabel("Four bridge total")
    plt.suptitle("Brooklyn, Manhattan, Queensboro")
    plt.show()
    
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
 
    # Train model and collect coefficients
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(X_train, y_train)
    print("Coefficients: High Temp: {:.4f}, Low Temp: {:.4f}, Precipitation: {:.4f}" .format(regr.coef_[0], regr.coef_[1], regr.coef_[2]))
    print("Intercept: {:.4f}" .format(regr.intercept_))
    
    # Calculate r2
    y_pred = regr.predict(X_test)
    r2 = metrics.r2_score(y_test, y_pred)
    print("r2: {:.4f}" .format(r2))

    
    print("\n\nApproach 2")
    r2 = 1
    coeff = []
    intercept = []

    for i in range(100):
        #y_pred = []
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True
        )
        regr = linear_model.LinearRegression(fit_intercept=True)
        regr.fit(X_train,y_train)
        y_pred = regr.predict(X_test)
        r2_test = metrics.r2_score(y_test, y_pred)
        if (r2_test < r2):
            r2 = r2_test
            coeff = regr.coef_
            intercept = regr.intercept_
    
    print("Coefficients: High Temp: {:.4f}, Low Temp: {:.4f}, Precipitation: {:.4f}" .format(coeff[0], coeff[1], coeff[2]))
    print("Intercept: {:.4f}" .format(intercept))
    print("r2: {:.4f}" .format(r2))


    print("\n\nApproach 3")
    lmbda = np.logspace(start=-1, stop=3, num=51, base=10.0)
    y = dataset_2['Total']
    x_high_temp = np.array(dataset_2['High Temp'])
    x_low_temp = np.array(dataset_2['Low Temp'])
    x_precip = np.array(dataset_2['Precipitation'])

    # Concatenate all data and reshape it
    X = np.column_stack((x_high_temp, x_low_temp, x_precip))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
 
    # Train model and collect coefficients
    maxr2 = 0
    finMSE = None
    model = None
    for l in lmbda:    
        regr = linear_model.Ridge(alpha = l, fit_intercept=True)
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        r2 = metrics.r2_score(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)

        if(r2 > maxr2):
            maxr2 = r2
            model = regr
            finMSE = mse

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
    axs[0,0].scatter(x_low_temp, y)
    axs[0,0].set_title("Low Temperature")
    axs[0,0].set_xlabel("Low Temperature")
    axs[0,0].set_ylabel("Total Bicycle Traffic")

    axs[0,1].scatter(x_high_temp, y)
    axs[0,1].set_title("High Temperature")
    axs[0,1].set_xlabel("High Temperature")
    axs[0,1].set_ylabel("Total Bicycle Traffic")

    axs[1,0].scatter(x_precip, y)
    axs[1,0].set_title("Precipitation")
    axs[1,0].set_xlabel("Precipitation")
    axs[1,0].set_ylabel("Total Bicycle Traffic")

    fig.delaxes(axs[1,1])
    fig.suptitle("Comparing Total Bicycle Traffic Against Low Temperature, High Temperature, and Precipitation")

    for i in axs:
        for j in i:
            j.grid(True)

    plt.show()

    print("Coefficients: High Temp: {:.4f}, Low Temp: {:.4f}, Precipitation: {:.4f}" .format(model.coef_[0], model.coef_[1], model.coef_[2]))
    print("Intercept: {:.4f}" .format(model.intercept_))
    
    print("r2: {:.4f}" .format(maxr2))
    print(f"MSE: {finMSE}")

    
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


question1()
question2()
question3()
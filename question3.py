import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path2.
'''
dataset_2 = pandas.read_csv('/home/shay/a/chand158/Python-Project/NYC_Bicycle_Counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Total']  = pandas.to_numeric(dataset_2['Total'].replace(',','', regex=True))
# print(dataset_2.to_string()) #This line will print out your data

# Problem 3: Classify what day is today based on number of bicyclists
def question3():
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

question3()
#Author: Manish Puri
#Note: Portions of the code have been completed using materials provided on HW assignment and discussions on Piazza


## Data and Visual Analytics - Homework 4
## Georgia Institute of Technology
## Applying ML algorithms to detect eye state

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn import grid_search
from sklearn.ensemble import RandomForestRegressor


######################################### Reading and Splitting the Data ###############################################
# XXX
# TODO: Read in all the data. Replace the 'xxx' with the path to the data set.
# XXX
data = pd.read_csv('eeg_dataset.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]

# The random state to use while splitting the data.
random_state = 100

# XXX
# TODO: Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
# Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 100.
# XXX
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.3,random_state=100)

# ############################################### Linear Regression ###################################################
# XXX
# TODO: Create a LinearRegression classifier and train it.
# XXX
lm = LinearRegression()

regout = lm.fit(x_train,y_train)
predictions = lm.predict(x_test)


#plt.scatter(y_test,predictions)

# XXX
# TODO: Test its accuracy (on the training set) using the accuracy_score method.
# TODO: Test its accuracy (on the testing set) using the accuracy_score method.
# Note: Round the output values greater than or equal to 0.5 to 1 and those less than 0.5 to 0. You can use y_predict.round() or any other method.
# XXX


print ("Train Accuracy  ", accuracy_score(y_train, lm.predict(x_train).round(), normalize=True))
print ("Test Accuracy ", accuracy_score(y_test, predictions.round(), normalize =True))


# ############################################### Random Forest Classifier ##############################################
# XXX
# TODO: Create a RandomForestClassifier and train it.
# XXX

def random_forest(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf


trained_model = random_forest(x_train, y_train)
print("trained model:: ", trained_model)
predictions = trained_model.predict(x_test)


# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX

print ("Training set accuracy  ", accuracy_score(y_train, trained_model.predict(x_train)))
print ("Testing set accuracy ", accuracy_score(y_test, predictions.round(), normalize =True))



# XXX
# TODO: Determine the feature importance as evaluated by the Random Forest Classifier.
#       Sort them in the descending order and print the feature numbers. The report the most important and the least important feature.
#       Mention the features with the exact names, e.g. X11, X1, etc.
#       Hint: There is a direct function available in sklearn to achieve this. Also checkout argsort() function in Python.
# XXX



randomClassify = RandomForestClassifier() 
randomClassify.fit(x_train, y_train) 
randomClassify.score(x_test, y_test)
feature_importance = pd.DataFrame(randomClassify.feature_importances_,
                                   index = x_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
print(feature_importance)




# XXX
# TODO: Tune the hyper-parameters 'n_estimators' and 'max_depth'.
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# Get the training and test set accuracy values after hyperparameter tuning.
# XXX



n_estimators = [int(x) for x in np.linspace(start=1, stop=10, num=2)]

print(n_estimators)
max_depth = [int(x) for x in np.linspace(10,40, num=2)]
print(max_depth)

random_grid = {'n_estimators': n_estimators,'max_depth': max_depth}

grid = GridSearchCV(estimator=randomClassify, param_grid=random_grid,scoring='roc_auc',verbose=1,n_jobs=-1)

grid_result = grid.fit(x_train, y_train)
print("best score: ", grid_result.best_score_)
print("best params: ", grid_result.best_params_)


print ("Training set accuracy  ", accuracy_score(y_train, grid_result.predict(x_train)))
print ("Testing set Accuracy ", accuracy_score(y_test, (grid_result.predict(x_test)).round(), normalize =True))



# ############################################ Support Vector Machine ###################################################
# XXX
# TODO: Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
# TODO: Create a SVC classifier and train it.
# XXX


scaler = StandardScaler().fit(x_data)
rescaledX = scaler.transform(x_data)

np.set_printoptions(precision=4)

X_train, X_test, y_train, y_test = train_test_split(rescaledX, y_data, test_size = 0.20)  
svclassifier = SVC()
svmd = svclassifier.fit(X_train, y_train)
y_pred = svmd.predict(X_test)


# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX

print ("Training set Accuracy  ", accuracy_score(y_train, svmd.predict(X_train)))
print ("Testing set Accuracy ", accuracy_score(y_test, y_pred.round(), normalize =True))



# XXX
# TODO: Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear).
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# Get the training and test set accuracy values after hyperparameter tuning.
# XXX



Cs = [1,10,100]
kernels = ['linear', 'rbf']

param_grid = {'C': Cs, 'kernel': kernels}

grid_search = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
grid_search.fit(rescaledX, y_data)
print("best params ", grid_search.best_params_)
print("best score ", grid_search.best_score_)


tuningpredict = grid_search.predict(X_test)

print ("Train Accuracy  ", accuracy_score(y_train, grid_search.predict(X_train)))
print ("Test Accuracy ", accuracy_score(y_test, tuningpredict.round(), normalize =True))


svclassifier2 = SVC(kernel='linear',C=1)
svmd2 = svclassifier2.fit(X_train, y_train)
y_pred2 = svmd2.predict(X_test)
print ("Train Accuracy  ", accuracy_score(y_train, svmd2.predict(X_train)))
print ("Test Accuracy ", accuracy_score(y_test, y_pred2.round(), normalize =True))


# XXX
# TODO: Calculate the mean training score, mean testing score and mean fit time for the 
# best combination of hyperparameter values that you obtained in Q3.2. The GridSearchCV 
# class holds a  ‘cv_results_’ dictionary that should help you report these metrics easily.
# XXX


print(grid_search.cv_results_['mean_train_score'][0])
print(grid_search.cv_results_['mean_test_score'][0])
print(grid_search.cv_results_['mean_fit_time'][0])


# ######################################### Principal Component Analysis #################################################
# XXX
# TODO: Perform dimensionality reduction of the data using PCA.
#       Set parameters n_component to 10 and svd_solver to 'full'. Keep other parameters at their default value.
#       Print the following arrays:
#       - Percentage of variance explained by each of the selected components
#       - The singular values corresponding to each of the selected components.
# XXX




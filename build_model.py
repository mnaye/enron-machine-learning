#For building models

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import sklearn.metrics as metric

def logit_pipeline (x_train, y_train, val, p, f):
    '''
    This method takes in training and testing data sets and pass them into a grid search pipeline with Logistic Regression pipeline
    :param x_train: training set
    :param y_train: class labels for training set
    :param val: boolean value for the usage of 10-fold cross validation
    :param p: boolean value for the usage of PCA
    :param f: boolean value for whether the data set has engineered features or not
    :return: the best parameters
    '''

    if f == True:
        feature_num = [10,12,14,16,18,20,22,24,26,27]

    if f == False:
        feature_num = [10,12,14,16,18]

    if p == True:
        steps = [('scaler', MinMaxScaler()),
                 ('decomposer', PCA()),
                 ('logistic_regression', LogisticRegression(random_state=42))]

        parameters = {
            'logistic_regression__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'logistic_regression__class_weight': [
                {True: 8, False: 1},
                {True: 10, False: 1},
                {True: 12, False: 1},
                'balanced'
            ]
        }

    if p == False:
        steps = [('scaler', MinMaxScaler()),
                 ('feature_selection', SelectKBest()),
                 ('logistic_regression', LogisticRegression(random_state=42))]

        parameters = {
            'feature_selection__k': feature_num,
            'logistic_regression__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'logistic_regression__class_weight': [
                {True: 8, False: 1},
                {True: 10, False: 1},
                {True: 12, False: 1},
                'balanced'
            ]
        }

    pipeline = Pipeline(steps)

    if val == True:
        cv = StratifiedShuffleSplit(n_splits=10, test_size= 0.2, train_size= 0.8, random_state=42)
        clf = GridSearchCV(pipeline, param_grid = parameters, cv=cv)
    if val == False:
        clf = GridSearchCV(pipeline, param_grid=parameters)

    clf.fit(x_train, y_train)
    return clf.best_estimator_


def rf_pipeline (x_train, y_train, val, p, f):

    '''
    This method takes training and testing data sets and passes them into a grid search pipeline using Random Forest
    :param x_train: training set
    :param y_train: class labels from training set
    :param val: boolean value for the usage of 10-fold cross validation
    :param p: boolean value for the usage of PCA
    :param f: boolean value for whether the data set has engineered features or not
    :return: the best parameters
    '''

    if f == True:
        feature_num = [10,12,14,16,18,20,22,24,26,27]

    if f == False:
        feature_num = [10, 12, 14, 16, 18]

    if p == True:
        steps = [('decomposer', PCA()),
                 ('random_forest', RandomForestClassifier(random_state=42))]

        parameters = {
            'random_forest__n_estimators': [10, 20, 30, 40, 50],
            'random_forest__min_samples_split': [2, 3, 4, 5, 10]
        }

    if p == False:
        steps = [('feature_selection', SelectKBest()),
                 ('random_forest', RandomForestClassifier(random_state=42))]

        parameters = {
            'feature_selection__k': feature_num,
            'random_forest__n_estimators': [10, 20, 30, 40, 50],
            'random_forest__min_samples_split': [2, 3, 4, 5, 10]
        }

    pipeline = Pipeline(steps)

    # scoring is focused on recall because we want to have a high recall rate to capture all true POI's for further investigation
    if val == True:
        cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, train_size=0.8,random_state=42)
        clf = GridSearchCV(pipeline, param_grid=parameters, cv=cv)
    if val == False:
        clf = GridSearchCV(pipeline, param_grid=parameters)

    clf.fit(x_train, y_train)
    return clf.best_estimator_



def show_model_metrics (name, best_params, x_test,y_test):
    '''
    This method makes predictions based on GridSearch CV pipeline's best parameters and return the model's metrics
    :param name: name of the model
    :param best_params: best parameters from GridSearch CV pipeline
    :param x_test: test data set
    :param y_test: test predictor
    :return: metric report
    '''
    predictions = best_params.predict(x_test)
    metric_report = metric.classification_report(y_test, predictions)
    print "Here are the results of ", name , " :\n"
    print metric_report
    print "---------------------------------------------------------------------------------------------------------"















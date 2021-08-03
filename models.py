# this file provides all models

class Classifier:

    """
    this class contains several classifiers with automatic parameter tuning 
    INPUTs
    Xtrain: train data
    ytrain: train labels
    Xtest: test data
    ytest: test labels
    """
    # necessary libraries
    from xgboost import XGBClassifier
    import lightgbm as lgb
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

    def __init__(self, Xtrain, ytrain, Xtest, ytest):

        # initializing class and feeding train test data and labels
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xtest = Xtest
        self.ytest = ytest

    def run_xgboost(self):
        pass

    def run_lgb(self):
        pass

    def run_svc(self):
        pass

    def run_rfc(self):
        pass

    def run_adaBoost(self):
        pass

    # https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
    # accuracy metrics: F1, ROC AUC
    # https://neptune.ai/blog/hyperparameter-tuning-in-python-a-complete-guide-2020

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
    # import lightgbm as lgb # has problem to load module
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    import optuna
    import sklearn.model_selection
    import xgboost as xgb
    from sklearn.metrics import f1_score
    import numpy as np
    import pdb

    def __init__(self, Xtrain, ytrain, Xtest, ytest):

        # initializing class and feeding train test data and labels
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xtest = Xtest
        self.ytest = ytest

    def run_xgboost(self, n_trials=100):

        # running xgb classifier and automatically optimizing paramters
        study = self.optuna.create_study(direction="maximize")
        study.optimize(self.xgb_objective, n_trials=n_trials, timeout=600)

        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    def run_lgb(self):
        # has problem to load module
        pass

    def run_svc(self):
        pass

    def run_rfc(self):
        pass

    def run_adaBoost(self):
        pass

    def xgb_objective(self, trial):

        # prepare train/test data and transoform them in xgb Matrix format
        dtrain = self.xgb.DMatrix(self.Xtrain, label=self.ytrain)
        dvalid = self.xgb.DMatrix(self.Xtest, label=self.ytest)

        # define parameter space
        param = {
            "verbosity": 0,
            "objective": "multi:softmax",
            # use exact for small dataset.
            "tree_method": "exact",
            # defines booster, gblinear for linear functions.
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            # L2 regularization weight.
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            # L1 regularization weight.
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            # sampling ratio for training data.
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            # evaluation metric
            "eval_metric": trial.suggest_categorical("eval_metric", ['merror', 'mlogloss']),
            # number of classes
            "num_class": len(self.np.unique(self.ytrain)),
        }

        if param["booster"] in ["gbtree", "dart"]:
            # maximum depth of the tree, signifies complexity of the tree.
            param["max_depth"] = trial.suggest_int("max_depth", 3, 50, step=2)
            # minimum child weight, larger the term more conservative the tree.
            param["min_child_weight"] = trial.suggest_int(
                "min_child_weight", 2, 20)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            # defines how selective algorithm is.
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"])

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical(
                "sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical(
                "normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float(
                "rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float(
                "skip_drop", 1e-8, 1.0, log=True)

        bst = self.xgb.train(param, dtrain)
        preds = bst.predict(dvalid)
        # self.pdb.set_trace()
        pred_labels = self.np.rint(preds)
        accuracy = self.f1_score(self.ytest, pred_labels, average='weighted')
        return accuracy

    # https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
    # accuracy metrics: F1, ROC AUC
    # https://neptune.ai/blog/hyperparameter-tuning-in-python-a-complete-guide-2020

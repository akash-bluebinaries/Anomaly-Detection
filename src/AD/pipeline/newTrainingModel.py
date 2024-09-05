import os,sys
import pandas as pd
import numpy as np
# from pycaret.classification import *
from imblearn.over_sampling import RandomOverSampler
import mlflow
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='_distutils_hack')

def load_data(path):
    data = pd.read_csv(path)
    return data

def data_cleaning(data):
    print("NA values available in data \n")
    print(data.isna().sum())
    data = data.dropna()
    print("After dropping NA values \n")
    print(data.isna().sum())
    return data

def data_preprocessing(data):
    final_data = data.drop('timestamps', axis=1)
#     cat_vars = []
#     for var in cat_vars:
#         cat_list = 'var'+'_'+var
#         cat_list = pd.get_dummies(data[var], prefix=var)
#         data1 = data.join(cat_list)
#         data = data1

#     cat_vars=[]
#     data_vars = data.columns.values.tolist()
#     to_keep = [i for i in data_vars if i not in cat_vars]
    
#     final_data = data[to_keep]

#     final_data.columns = final_data.columns.str.replace('.','_')
#     final_data.columns = final_data.columns.str.replace(' ','_')
    return final_data


def train_test_split(final_data):
    from sklearn.model_selection import train_test_split
    X = final_data.drop('Anomaly', axis =1 )
    y = final_data['Anomaly']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=123)
    return X_train, X_test, y_train, y_test


def over_sampling_target_class(X_train,y_train):
    from imblearn.over_sampling import RandomOverSampler
    
    sampler = RandomOverSampler()
    # column = X_train.columns
    os_X, os_y = sampler.fit_resample(X_train, y_train)

    os_X = pd.DataFrame(data=os_X, columns=X_train.columns )
    os_y = pd.DataFrame(data=os_y, columns=['Anomaly'])

    print("Length of oversampled data is ", len(os_X))
    print("Number of Anomaly in oversampled data is ",len(os_y[os_y['Anomaly']==1]))
    print("Number of Not Anomaly in oversampled data is ",len(os_y[os_y['Anomaly']==0]))

    X_train = os_X
    y_train = os_y['Anomaly']

    return X_train, y_train

def train_basic_classifier(X_train,y_train):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def predict_on_test(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def predict_prob_on_test(model, X_test):
    y_pred = model.predict_proba(X_test)
    return y_pred

def get_metrics(y_true, y_pred, y_pred_prob):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    entropy = log_loss(y_true, y_pred)
    return {'accuracy': round(accuracy,2), 'precision':round(precision,2), 'recall':round(recall,2),'entropy':round(entropy,2)}

def create_roc_auc_plot(clf, X, y):
    import matplotlib.pyplot as plt
    from sklearn import metrics
    metrics.plot_roc_curve(clf, X,y)
    plt.savefig('roc_auc_curve.png')

def create_confusion_matrix_plot(clf, X, y):
    from sklearn.metrics import plot_confusion_matrix
    import matplotlib.pyplot as plt
    plot_confusion_matrix(clf, X, y)
    plt.savefig('confusion_matrix.png')


def hyper_parameter_tuning(X_train, y_train):
    n_estimators = [5,21,51,101]
    max_features = ['auto','sqrt']
    max_depth = [int(x) for x in np.linspace(10,120, num=12)]
    min_samples_split = [2, 6, 10]
    min_samples_leaf = [1, 3, 5]
    bootstrap = [True,False]

    random_grid = {'n_estimators':n_estimators,
                   'max_features':max_features,
                   'max_depth':max_depth,
                   'min_samples_split':min_samples_split,
                   'min_samples_leaf':min_samples_leaf,
                   'bootstrap':bootstrap
                   }
    
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier

    classifier = RandomForestClassifier()
    model_tuning = RandomizedSearchCV(estimator=classifier, param_distributions=random_grid,
                                      n_iter=10, cv=2, verbose=2, random_state=35)
    
    model_tuning.fit(X_train, y_train)

    print('Random grid:', random_grid,'\n')
    print('Best Parameters:', model_tuning.best_params_,'\n')


    best_params = model_tuning.best_params_

    n_estimators = best_params['n_estimators']
    max_features = best_params['max_features']
    max_depth = best_params['max_depth']
    min_samples_split = best_params['min_samples_split']
    min_samples_leaf = best_params['min_samples_leaf']
    bootstrap = best_params['bootstrap']

    model_tuned = RandomForestClassifier(n_estimators=n_estimators,
                                         max_features=max_features,
                                         max_depth=max_depth,
                                         min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf,
                                         bootstrap=bootstrap
                                         )
    
    model_tuned.fit(X_train, y_train)
    return model_tuned, best_params


data = pd.read_csv('D:/bmsAnomalyDetection/application/Anomaly-detection/notebook/data/labelled_data.csv')
cleaned_data = data_cleaning(data)
final_data = data_preprocessing(cleaned_data)

X_train, X_test, y_train, y_test = train_test_split(final_data)

X_train, y_train = over_sampling_target_class(X_train, y_train)

clf = train_basic_classifier(X_train, y_train)

y_pred = predict_on_test(clf, X_test)

y_pred_prob = clf.predict_proba(X_test)[:,1]

run_metrics = get_metrics(y_test, y_pred,y_pred_prob)

create_roc_auc_plot(clf, X_test, y_test)

create_confusion_matrix_plot(clf, X_test,y_test)




# MLFLOW
################################################################
experiment_name = "Experiment_1"
run_name = "Basic_Model"
# run_metrics = get_metrics(y_test, y_pred, y_pred_prob)
# print(run_metrics)


def create_experiment(experiment_name, run_name, run_metrics,
                      model, confusion_matrix_path=None, 
                      roc_auc_curve_path=None, run_params=None):

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        if not run_params == None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])

        for metric in run_metrics:
            mlflow.log_metric(metric, run_metrics[metric])

        mlflow.sklearn.log_model(model,"model")

        if not confusion_matrix_path == None:
            mlflow.log_artifact(confusion_matrix_path, "confusion_matrix")

        if not roc_auc_curve_path == None:
            mlflow.log_artifact(roc_auc_curve_path, "roc_auc_curve")

        mlflow.set_tag("tag1","Random Forest")
        mlflow.set_tags({"tag2":"Randomized Search CV", "tag3":"Production"})

    print(f"Run - %s is logged to Experiment - %s" %(run_name, experiment_name))
    mlflow.end_run()
################################################################################



try:
    create_experiment(experiment_name,run_name,run_metrics,clf,'confusion_matrix.png','roc_auc_curve.png')
except Exception as e:
    print(f"MLflow logging failed: {str(e)}")



# Another experiment with tuned model

experiment_name = "Experiment_2"
run_name = "Tuned_Model"

model_tuned, best_params = hyper_parameter_tuning(X_train, y_train)
run_params = best_params

y_pred = predict_on_test(model_tuned,X_test)
y_pred_prob = predict_prob_on_test(model_tuned,X_test)
run_metrics = get_metrics(y_test,y_pred,y_pred_prob)

for param in run_params:
    print(param, run_params[param])


create_experiment(experiment_name,run_name,run_metrics,model_tuned,'confusion_matrix.png','roc_auc_curve.png',run_params)
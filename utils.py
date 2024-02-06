import pandas as pd
import os
from sklearn.metrics import classification_report
# load data
import os
import pandas as pd
import numpy as np
# test models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import time
# feature importance plot
import matplotlib.pyplot as plt


def linearize(u):
    """
    Fonction pour mettre en 1 seule lignes le tableau de statistiques pd.describe()
    """
    all=[]
    for line in range(len(u)):
        all.append(u.iloc[line])
    return pd.concat(all, axis=0).T

def load_data(directory = 'data/group3/config_1', drop_col='', drop_feat=''):

    """
    Fonction pour charger les donnée:
    directory : str() chemin d'acces
    drop_col : list() nom des statistiques à ne pas utiliser (pd.describe())
    drop_feat : list() nom des features à ne pas utiliser

    return : pd.DataFrame()
    """

    all_data=[]
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        data = pd.read_csv(f)
        if drop_col:
            data.drop(columns=drop_col, inplace=True)
        u = data.describe().T
        if drop_feat:
            u.drop(columns=drop_feat, inplace=True)
        number = pd.DataFrame(linearize(u)).T
        # add labels
        if directory == 'data/config1':
            number['label'] = int(filename.split('_')[1])
        else:
            number['label'] = int(filename[0])
        all_data.append(number)

    to_test = pd.concat(all_data)

    new_col=[]
    for stat_ in u.columns:
        for feat_ in data.columns:
            new_col.append(stat_.upper()+"_"+feat_)
    new_col.append('label')
    to_test.columns=new_col

    return to_test

to_test = load_data() # drop_col='t', drop_feat='count')
to_test.shape



def test_models(to_test=to_test):
    """
    Fonction permettant de tester 4 modèles pour la classification avec 10 split de cross-validation
    to_test : pd.DataFrame()
    """

    sc = StandardScaler()
    X=sc.fit_transform(to_test.drop("label", axis=1))
    y=to_test["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    models=[
        LinearRegression(),
        LogisticRegression(solver='liblinear'),
        DecisionTreeClassifier(),
        RandomForestClassifier()
    ]

    for model in models:
        cv =cross_validate(model,X_train,y_train,cv=10)
        print(f"{model} score :{round(cv['test_score'].mean(),2)}, time {round(cv['score_time'].mean(),4)}")
    return X, y, X_train, X_test, y_train, y_test




def optimize_logreg(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test):
    logreg_model = LogisticRegression(multi_class='auto', solver='liblinear')

    param_grid = {
        'C': np.logspace(-4, 4, 40),  
        'penalty': ['l1', 'l2'], 
        'max_iter': [100, 500, 1000, 10000],  
    }

    random_search = RandomizedSearchCV(logreg_model, 
                                    param_distributions=param_grid, 
                                    n_iter=10, 
                                    scoring='accuracy', 
                                    cv=5, 
                                    random_state=42,
                                    refit=True)

    random_search.fit(X_train, y_train)
    t0 = time.time()
    y_pred = random_search.best_estimator_.predict(X_test)
    t_fit = time.time()-t0
    print(classification_report(y_test,y_pred))
    print("fit time :",round(t_fit,4))
    return random_search


def feat_importance_logreg(random_search=random_search, to_test=to_test, show=False):
    # fit the model with best params from random search
    p_ = random_search.best_params_
    model_lr = LogisticRegression(**p_,solver='liblinear')
    model_lr.fit(X, y)

    if show:
        label_correlations = to_test.corr()[-1:].iloc[:,:-1]
        # extract feature importance
        coefficients = model_lr.coef_[0]
        feature_importance = pd.DataFrame({'Feature': to_test.columns[:-1], 'Importance': np.abs(coefficients)})
        n_stats = feature_importance.Feature.str.startswith('MAX').sum()
        stats = ([a.split('_')[0] for a in feature_importance[::n_stats].Feature])
        r={}
        for i, stat_ in enumerate(stats):
            r[stat_] = feature_importance.iloc[i*n_stats:(i+1)*n_stats,1:].values.reshape(n_stats)
        r = pd.DataFrame(r)
        r.index=['-'.join(a.split('_')[1:]) for a in label_correlations.columns[:n_stats].values]
        sns.heatmap(r*10, annot=True, fmt='.1f')
        plt.title(f"abs(Variable importance)*10^1 pour {model_lr}")
        plt.show()
    return model_lr

def optimize_rfc(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test):
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
    }

    random_search = RandomizedSearchCV(RandomForestClassifier(), 
                                    param_distributions=param_grid, 
                                    n_iter=10, 
                                    scoring='accuracy', 
                                    cv=5, 
                                    random_state=42, 
                                    refit=True)
    random_search.fit(X_train, y_train)

    start_time = time.time()
    y_pred = random_search.best_estimator_.predict(X_test)
    end_time = time.time()

    print("best estimator",random_search.best_estimator_)
    print(classification_report(y_test,y_pred))
    print("time to predict :",round(end_time - start_time,4))
    return random_search



def feat_importance_rfc(random_search=random_search, to_test=to_test, show=False):
    # fit the model with best params from random search
    p_ = random_search.best_params_
    model_rfc = RandomForestClassifier(**p_)
    model_rfc.fit(X, y)

    if show:
        label_correlations = to_test.corr()[-1:].iloc[:,:-1]
        # extract feature importance
        coefficients = model_rfc.feature_importances_
        feature_importance = pd.DataFrame({'Feature': label_correlations.columns, 'Importance': np.abs(coefficients)})
        n_stats = feature_importance.Feature.str.startswith('MAX').sum()
        stats = ([a.split('_')[0] for a in feature_importance[::n_stats].Feature])
        r={}
        for i, stat_ in enumerate(stats):
            r[stat_] = feature_importance.iloc[i*n_stats:(i+1)*n_stats,1:].values.reshape(n_stats)
        s=r
        r = pd.DataFrame(r)
        r.index=['-'.join(a.split('_')[1:]) for a in label_correlations.columns[:n_stats].values]
        sns.heatmap(r*100, annot=True, fmt='.1f')
        plt.title(f"abs(Variable importance)*10^2 pour {model_rfc}")
        plt.show()

    return model_rfc
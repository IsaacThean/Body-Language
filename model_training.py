import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle


def readCSV(file):
    df = pd.read_csv(file)
    return df

def makeTrain(df):
    y = df["class"]
    X = df.drop('class', axis=1) # features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    return X_train,X_test,y_train,y_test

def fitModels(data):

    X_train, X_test, y_train, y_test = data
 
    pipelines = {
        'lr':make_pipeline(StandardScaler(), LogisticRegression()),
        'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
        'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
        'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    }

    fit_models = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algo] = model

    for algo, model in fit_models.items():
        yhat = model.predict(X_test)
        print(algo, accuracy_score(y_test, yhat))

    fit_models['rf'].predict(X_test)

    with open('body_language.pkl', 'wb') as f:
        pickle.dump(fit_models['rf'], f)

fitModels(makeTrain(readCSV("landmarks.csv")))
    
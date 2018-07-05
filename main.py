import pandas as pd
import matplotlib.pyplot as plt
from numpy import nanmedian, NaN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np

# load the train file
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

def preprocess(df):
    df.drop(['Name', 'Ticket'], axis=1, inplace=True)
    X = df
    
    X['Sex'] = (X['Sex'].replace('male',1)
                    .replace('female',0))
    X['Embarked'] = (X['Embarked'].replace('C',0)
                        .replace('Q',1)
                        .replace('S',2))
    X['Cabin'] = (X['Cabin'].fillna(value = '0' )
                            .apply(lambda x: (x).split(' ')[0][0])
                            .replace({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8})
                            .replace('0', np.nan))
    return X
    
def fillnan(df):
    cols = df.columns
    for col in cols:
        median = nanmedian(df[col])
        df[col] = [median if np.isnan(val) else int(val) for val in df[col]]
    return df

def scorer(classifier,X,Y):
    return cross_val_score(classifier, X, Y, cv=5).mean()

X = fillnan(preprocess(df))
X = df.drop(['Survived','PassengerId','SibSp'], axis=1)
X_test = fillnan(preprocess(df_test)).drop('SibSp', axis=1)
Y = df['Survived']
#split train and test data


#create a Random Forest Classifier
classifierRF = RandomForestClassifier(n_estimators=350,min_samples_leaf=3,min_samples_split=10)
classifierRF.fit(X, Y)
scoreRF = scorer(classifierRF,X,Y)

#create a GradienBoostingClassifier
classifierGB = GradientBoostingClassifier(n_estimators=350,min_samples_leaf=10,min_samples_split=2, loss="exponential")
classifierGB.fit(X, Y)
scoreGB = scorer(classifierGB,X,Y)

#create an SVM classifier
classifierSVM = svm.SVC()
classifierSVM.fit(X, Y)
scoreSVM = scorer(classifierSVM,X,Y)

#create a Logistic Regression Model
classifierLR = LogisticRegression(tol=0.00005, C=2, dual=False,fit_intercept=True, intercept_scaling=1, max_iter = 200)
classifierLR.fit(X,Y)
scoreLR = scorer(classifierLR,X,Y)
#Fit the random forest classifier with the train data
ints = range(0,4)
labels = ['SVM','LR','GB','RF']
plt.figure(figsize=(12,4))
plt.plot(ints,[scoreSVM*100, scoreLR*100, scoreGB*100, scoreRF*100],'ro-')
plt.xlim([-1,4])
plt.xticks(ints, labels, rotation='horizontal')
print "Ranodm Forest Model Score: ", scoreRF
print "Gradient Boosting Model Score: ", scoreGB
print "SVM Model Score: ", scoreSVM
print "Logistic regression Model Score: ", scoreLR
#X_test['Survived'] = classifierRF.predict(X_test.drop('PassengerId', axis=1))
#X_test = X_test[['PassengerId','Survived']]
#X_test.to_csv('submission2.csv',delimiter=',',index=False, header=True)
'''param_grid = {"max_depth": [3, None],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "n_estimators":[100,500,1000]
              }

# run grid search
grid_search = GridSearchCV(classifier, param_grid=param_grid, n_jobs=-1)
grid_search.fit(X, Y)'''
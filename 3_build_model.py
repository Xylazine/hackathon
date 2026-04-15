from random import Random

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

df = pd.read_csv("clean_features.csv")

X = df.iloc[:,1:-1]   # predictor columns
y = df.iloc[:, -1]   # target column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
SS = StandardScaler()
SS.fit(X_train)   # fit scalar
X_train = pd.DataFrame(SS.transform(X_train), columns=X_train.columns)   # scale training data
X_test = pd.DataFrame(SS.transform(X_test), columns=X_test.columns)   # scale testing data
kf = KFold(n_splits = 5, shuffle = True, random_state=123)   # cross-validation


# SVM
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_pred = svc_model.predict(X_test)

base_accuracy = accuracy_score(y_test, y_pred)
print(base_accuracy)



# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=123)
rf_model.fit(X_train, y_train)
y_pred = rf_model
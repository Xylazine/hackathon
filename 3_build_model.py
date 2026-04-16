import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

df = pd.read_csv("cell_segmentation_classifier/clean_features.csv")

X = df.iloc[:,1:-1]   # predictor columns
y = df.iloc[:, -1]   # target column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
SS = StandardScaler()
SS.fit(X_train)   # fit scalar
X_train = pd.DataFrame(SS.transform(X_train), columns=X_train.columns)   # scale training data
X_test = pd.DataFrame(SS.transform(X_test), columns=X_test.columns)   # scale testing data
kf = KFold(n_splits = 5, shuffle = True, random_state=123)   # cross-validation


## SVM

# base model
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_pred = svc_model.predict(X_test)

svc_base_accuracy = np.mean(cross_val_score(svc_model, X_train, y_train, cv=kf))
print("SVC base accuracy:", round(svc_base_accuracy, 3))

df_svc = pd.DataFrame({
    "Predicted_preTune":  y_pred,
    "Actual": y_test
})

# tune model
Cs = [10, 100, 200]
gammas = ['scale', 0.01, .1, 1]
kernels = ['rbf', 'linear']
# set starting parameters to the defaults
best_C = 1.0
best_gamma = 'scale'
best_kernel = 'rbf'
best_accuracy = svc_base_accuracy

for c in Cs:
  for g in gammas:
    for k in kernels:
      model = SVC(C=c, gamma=g, kernel=k)
      accuracies = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy")
      print("C:", c, "gamme:", g, "kernel:", k, "accuracy:", round(accuracies.mean(), 3))
      mean_accuracy = round(np.mean(accuracies), 3)
      if mean_accuracy >= best_accuracy:
        best_accuracy = mean_accuracy
        best_C = c
        best_gamma = g
        best_kernel = k

# build and evaluate final model
svc_model = SVC(C=best_C, gamma=best_gamma, kernel=best_kernel)
svc_model.fit(X_train, y_train)
svc_pred = svc_model.predict(X_test)
svc_score = round(accuracy_score(X_test, y_test), 3)
print("SVC model after tuning:", "\nAccuracy: ", svc_score, "\nBest C: ", best_C,
      "\nBest gamma: ", best_gamma, "\nBest kernel: ", best_kernel)

# upload predictions
df_svc["Predicted_tuned"] = svc_pred
df_svc.to_csv('svc_predictions.csv')






## Random Forest

# base model
rf_model = RandomForestClassifier(n_estimators=100, random_state=123)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
rf_base_accuracy = np.mean(cross_val_score(rf_model, X_train, y_train, cv=kf))
print("Random Forest base accuracy:", round(rf_base_accuracy, 3))

df_rf = pd.DataFrame({
    "Predicted_preTune":  y_pred,
    "Actual": y_test
})

# tune model
estimators = [100, 150, 200]
splits = [3, 4, 5]
leaves = [2, 3, 4]
# set starting parameters to the defaults
best_est = 100
best_split = 2
best_leaf = 1
best_accuracy = rf_base_accuracy

for e in estimators:
  for s in splits:
    for l in leaves:
      model = RandomForestClassifier(n_estimators=e,
                          min_samples_split=s,
                          min_samples_leaf=l)
      accuracies = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
      print("estimators:", e,"splits", "leaves:", l, "accuracy:", round(accuracies.mean(), 3))
      mean_accuracies = np.mean(accuracies)
      if mean_accuracies >= best_accuracy:
        best_accuracy = mean_accuracies
        best_est = e
        best_split = s
        best_leaf = l

# build and evaluate final model
rf_model = RandomForestClassifier(n_estimators=best_est, min_samples_split=best_split, min_samples_leaf=best_leaf)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_score = round(accuracy_score(X_test, y_test), 3)
print("RF model after tuning:", "\nBest score: ", rf_score, "\nBest # estimators: ",
      best_est, "\nBest # splits: ", best_split, "\nBest leaf: ", best_leaf)

# upload data
df_rf["Predicted_tuned"] = rf_pred
df_rf.to_csv('rf_predictions.csv')
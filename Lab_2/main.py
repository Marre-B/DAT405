import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sk
import sklearn.model_selection as skm
import sklearn.linear_model as skl
import sklearn.neighbors as skn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

## Problem 1 ---------------------------------------------------------------

# Load data
data = pd.read_csv('data_assignment2.csv')

area = data['Living_area']
price = data['Selling_price']

# Use NumPy to fit a linear regression model
coef = np.polyfit(area, price, 1)
poly1d_fn = np.poly1d(coef)

# Find slope and intercept
slope, intercept = coef

print("Slope: " + str(slope))
print("Intercept: " + str(intercept))

# Predict price for 100, 150, 200 sqm
print("100m^2 price: " + str(poly1d_fn(100)))
print("150m^2 price: " + str(poly1d_fn(150)))
print("200m^2 price: " + str(poly1d_fn(200)))

# Plot graph
plt.plot(area, price, 'yo', area, poly1d_fn(area), '--k')
plt.xlabel('Living area')
plt.ylabel('Selling price')
plt.title('Living area vs Selling price')
plt.show()

# Plot residual graph
diff = poly1d_fn(area) - price
plt.plot(area, diff, 'or')
plt.show()


## Problem 2 ---------------------------------------------------------------

def to_target(x):
    return list(dataset.target_names)[x]

# Load data
dataset = load_iris()
df = pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['target'] = pd.Series(dataset.target)
df['target_names'] = df['target'].apply(to_target)

# Define predictor and predicted datasets
X = df.drop(['target','target_names'], axis=1).values
y = df['target_names'].values

# split taining and test set
X_train, X_test, y_train, y_test = skm.train_test_split(X, y)

# train the model at logistic regression
model_lr = LogisticRegression().fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# train the model at KNN
model_knn = skn.KNeighborsClassifier(n_neighbors=100, weights="distance").fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)

# compute accuracy of the model
print("LR model accurcy: " + str(model_lr.score(X_test, y_test)))
print("KNN model accuracy: " + str(model_knn.score(X_test, y_test)))

arr = []
for x in range(1,100):
    model_knn = skn.KNeighborsClassifier(n_neighbors=x, weights="distance").fit(X_train, y_train)
    y_pred_knn = model_knn.predict(X_test)
    temp = model_knn.score(X_test, y_test)
    print("KNN model accuracy for " + str(x) + " neighbors: " + str(temp))
    arr.append(temp)

coef = np.polyfit(range(1, 100), arr, 1)
poly1d_fn = np.poly1d(coef)

plt.plot(range(1,100), arr, 'yo', arr, poly1d_fn(arr), '--k')
plt.xlabel('Amount of neighbors')
plt.ylabel('Accuracy')
plt.title('Amount of neighbors vs Accuracy')
plt.show()

# Plot confusion matrix
confusion_matrix = sk.confusion_matrix(y_test, y_pred_knn)
cm_display = sk.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)

cm_display.plot()
plt.show()

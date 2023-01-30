import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sk
import sklearn.model_selection as skm
import sklearn.linear_model as skl
import sklearn.neighbors as skn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

## Problem 1

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

## Problem 2

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

# train the model
model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# compute accuracy of the model
print(model.score(X_test, y_test))

# Plot confusion matrix
confusion_matrix = sk.confusion_matrix(y_test, y_pred)
cm_display = sk.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)

cm_display.plot()
plt.show()
# Write your pre-processing code here

import pandas as pd
import os
import sklearn.model_selection as skm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline


# ------------------ Question 1 ------------------

# Read data from folder and sort it into a dataframe
def readData(folder):
    files = os.listdir(folder) # 'easy_ham'
    df = pd.DataFrame(columns=['file_name', 'data', 'label'])

    for i, file in enumerate(files):
        with open(folder + '/' + file, 'r', encoding="latin-1") as f:
            temp_df = pd.DataFrame({'file_name': file, 'data': f.read(), 'label': folder}, index=[i])
            df = pd.concat([df, temp_df])
    return df

# Read data
spam = readData('spam')
ham = readData('easy_ham')
hard_ham = readData('hard_ham')

df = pd.concat([spam, ham])

# Split data into training and testing
X_train, X_test, y_train, y_test = skm.train_test_split(df['data'], df['label'], test_size=0.1)

# ------------------ Question 2 ------------------

model_Multi = make_pipeline(TfidfVectorizer(), MultinomialNB())
model_Bern = make_pipeline(TfidfVectorizer(), BernoulliNB())

model_Multi.fit(X_train, y_train)
model_Bern.fit(X_train, y_train)

y_pred_Multi = model_Multi.predict(X_test)
y_pred_Bern = model_Bern.predict(X_test)

print("Accuracy Multi: ", accuracy_score(y_test, y_pred_Multi))
print("Accuracy Bern: ", accuracy_score(y_test, y_pred_Bern))

# Find the confusion matrix
tn1, fp1, fn1, tp1 = confusion_matrix(y_test, y_pred_Multi).ravel()
tn2, fp2, fn2, tp2 = confusion_matrix(y_test, y_pred_Bern).ravel()

# Print the confusion matrix
print("True Positive Multi: " + str(tn1/(tn1+fp1)))
print("True Negative Multi: " + str(tp1/(tp1+fn1)))
print("True Positive Bern: " + str(tn2/(tn2+fp2)))
print("True Negative Bern: " + str(tp2/(tp2+fn2)))

# ------------------ Question 3 ------------------







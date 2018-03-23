import pandas as pd
import numpy as np
import random
import csv
import statsmodels as sm
import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics
import sklearn.tree as tree
import seaborn as sns

"""
original_data = pd.read_csv(
    "./datasets/adult.data",
    names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
#original_data.tail()

def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders

encoded_data, _ = number_encode_features(original_data)
sns.heatmap(encoded_data.corr(), square=True)
plt.show()
"""

with open("./datasets/adult.csv", 'rb') as file:
    data = list()
    for row in file:
        data.append(str(row).split('\n'))
file.close()

random.shuffle(data)

train_data = data[:int((len(data)+1)*.20)]
test_data = data[int((len(data)+1)*.95+1):]
train_data = list(train_data)
test_data = list(test_data)

writer_train = csv.writer(open("./datasets/adult_train.csv", 'w'))
writer_test = csv.writer(open("./datasets/adult_test.csv", 'w'))
for row in train_data:
    writer_train.writerow(row)
for row in test_data:
    writer_test.writerow(row)

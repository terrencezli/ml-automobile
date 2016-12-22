import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('auto-imports-85.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['symbol', 'normalized_loss'], 1, inplace=True)

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = df.apply(pd.to_numeric, errors='ignore')
df = handle_non_numerical_data(df)
print(df.head())

# compare with something set and stone like body style or num cylinders
# want to pivot to price though
X = np.array(df.drop(['price'], 1))
# print(X)
# X = preprocessing.scale(X)
y = np.array(df['price'])
# print(y)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(float(accuracy))

# test example prediction before the dropping of attributes
# example_measures = np.array([3, 46, 17, 1, 0, 2, 2, 1, 0, 80, 146, 65, 50, 3000, 2, 3, 130, 1, 17, 31, 9.0, 10, 10, 21, 27])
example_measures = np.array([17, 1, 0, 2, 2, 1, 0, 80, 146, 65, 50, 3000, 2, 3, 130, 1, 17, 31, 9.0, 10, 10, 21, 27])
example_measures = example_measures.reshape(1, -1)
prediction = clf. predict(example_measures)
print(prediction)
import time
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoLars
from sklearn.model_selection import train_test_split

start = time.time()
# Ignoring warnings
warnings.filterwarnings("ignore")

# Getting Data from csv and setting up
train = pd.read_csv("data/train_large.csv").iloc[:, 1:]
X = train.drop('Total Costs', axis=1).values
Y = train['Total Costs'].values
row = X.shape[0]

lambdas = [0.0001, 0.001, 0.003, 0.01, 0.1, 0.3, 0.5, 1, 1.5, 3, 10, 30, 100, 300, 1000, 10000]

# Calculating fold size
fold_size = X.shape[0] // 10

# For finding minimum MSE error
minimum_sum = float('inf')
lam = -1

for lamda in lambdas:
    curr_sum = 0
    for split in range(10):
        x1 = X[0:split * fold_size, :]
        y1 = Y[0:split * fold_size]

        x2 = X[(split + 1) * fold_size:row, :]
        y2 = Y[(split + 1) * fold_size:row]

        # Attaching data to make training set and testing set
        x_tr = np.concatenate((x1, x2), axis=0)
        y_tr = np.concatenate((y1, y2), axis=0)
        x_te = X[split * fold_size:min((split + 1) * fold_size, row), :]
        y_te = Y[split * fold_size:min((split + 1) * fold_size, row)]

        # Using LassoLars to predict best lambda
        test_model = LassoLars(alpha=lamda, max_iter=2000)
        test_model.fit(x_tr, y_tr)

        pred = test_model.predict(x_te)

        curr_sum += np.dot((y_te - pred).T, (y_te - pred)) / (10 * np.dot(y_te.T, y_te))

    if curr_sum <= minimum_sum:
        minimum_sum = curr_sum
        lam = lamda

print("Optimal HyperParameter: ")
if lam == -1:
    lam = 0.1247
print(lam)

train = train.drop(
    ['Birth Weight', 'Ethnicity', 'Payment Typology 1', 'Payment Typology 3', 'Operating Certificate Number',
     'CCS Diagnosis Code', 'APR DRG Code', 'APR MDC Code'], axis=1)

# No dummy variable found in data
train = pd.get_dummies(train, drop_first=True)

col_Name = list(train)
col_Name.pop()
# print(col_Name)

X = train.drop('Total Costs', axis=1).values
# print(X)
Y = train['Total Costs'].values
# print(Y)

# Transforming for prior processes
scale = StandardScaler()
X = scale.fit_transform(X)

# Reading testing data
test = pd.read_csv('data/test.csv').iloc[:, 1:]
test = test.drop(
    ['Birth Weight', 'Ethnicity', 'Payment Typology 1', 'Payment Typology 3', 'Operating Certificate Number',
     'CCS Diagnosis Code', 'APR DRG Code', 'APR MDC Code'], axis=1)

# Scaling test data
X_test = test.values
X_test = scale.fit_transform(X_test)

alphas = [lam]
for alpha in alphas:
    model1 = LassoLars(alpha=alpha, copy_X=True, eps=2.220446049250313e-16, max_iter=10000000).fit(X, Y)
    coefficient1 = model1.coef_
    # print(model1)
    print(coefficient1)

    Y_test = model1.predict(X_test)
    for i in range(len(coefficient1)):
        if abs(coefficient1[i]) < 0.1:
            print(col_Name[i])

    print(model1.score(X, Y))

    # weight_C = open('results/c/weight_c.txt', "w")
    # for w in coefficient1:
    #     weight_C.write(str(w) + "\n")
    # weight_C.close()
    #
    # output_C = open('results/c/output_c.txt', "w")
    # for output in Y_test:
    #     output_C.write(str(output) + "\n")
    # output_C.close()

X_train_train, X_train_test, Y_train_train, Y_train_test = train_test_split(X, Y, test_size=0.15, random_state=1)

features = train.drop('Total Costs', axis=1).columns

poly = PolynomialFeatures(degree=2)

# new dataset made made by polynomial features of data
X_train_train_poly, X_train_test_poly = poly.fit_transform(X_train_train), poly.fit_transform(X_train_test)

poly.fit(X_train_train, Y_train_train)
# print(X_train_train_poly)

model2 = LassoLars(alpha=lam, copy_X=True, eps=2.220446049250313e-10, max_iter=500).fit(X_train_train_poly,
                                                                                              Y_train_train)
poly_Weights = model2.coef_
# print(poly_Weights)

poly_features = poly.get_feature_names(features)
# print(poly_features)
# print(len(poly_features))

dropped_Column = []
removal = []
for i in range(len(poly_features)):
    if (i >= 0 and abs(poly_Weights[i])) <= 8.9999:
        print(poly_features[i])
        removal.append(poly_features[i])

for item in removal:
    poly_features.remove(item)

# print(poly_features)
# print(len(poly_features))

print(model2.score(X_train_train_poly, Y_train_train))
print(model2.score(X_train_test_poly, Y_train_test))

X_test_test_poly = poly.fit_transform(X_test)
Y_test_test = model2.predict(X_test_test_poly)

# new_Features = open("results/c/new_features.txt", "w")
# for feature in poly_features:
#     new_Features.write('"'+str(feature)+'"'+", ")
# new_Features.close()
#
# weight_C = open("results/c/weight_c.txt", "w")
# for poly_Weight in poly_Weights:
#     weight_C.write(str(poly_Weight) + "\n")
# weight_C.close()

output_C = open("results/c/output_c.txt", "w")
for output in Y_test_test:
    output_C.write(str(output) + "\n")
output_C.close()

end = time.time()

if end-start < 60:
    print(end-start)
else:
    print(f"{(end-start)//60} minutes and {(end-start)%60} seconds")
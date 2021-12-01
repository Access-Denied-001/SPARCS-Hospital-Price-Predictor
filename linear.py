import sys
import time
import numpy as np
import pandas as pd

# used in only 3rd part
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoLars


def lin_reg(trainfile, testfile, outputfile, weightfile):
    start = time.time()
    # reading csv from training data
    train = pd.read_csv(trainfile).iloc[:, 1:]

    # Adjusting feature vector according to my model
    X_train = train.iloc[:, 0:train.shape[1] - 1].values
    X_train = np.insert(X_train, 0, 1, axis=1)

    # print(pd.DataFrame(X_train))

    # Reading target values
    Y_train = train.iloc[:, train.shape[1] - 1].values

    # print(pd.DataFrame(Y_train))

    # Using Moore Penrose Inverse
    # W = (XTX)^(-1)(XTY)
    X_train_transpose = X_train.transpose()
    W = np.dot(np.linalg.inv(np.dot(X_train_transpose, X_train)), np.dot(X_train_transpose, Y_train))
    # print(pd.DataFrame(W))

    # Reading testing data
    test = pd.read_csv(testfile).iloc[:, 1:]
    # Testing feature vector
    X_test = test.iloc[:, 0:].values
    X_test = np.insert(X_test, 0, 1, axis=1)
    # print(pd.DataFrame(X_test))

    # Predicting target values XW
    my_output = np.dot(X_test, W)
    # print(pd.DataFrame(my_output))

    try:
        # Writing Predictions and Model Parameters
        W_File = open(weightfile, "w")
        for weight in W:
            W_File.write(str(weight) + "\n")
        W_File.close()

        Output_File = open(outputfile, "w")
        for output in my_output:
            Output_File.write(str(output) + "\n")
        Output_File.close()
    except:
        # print(my_output)
        print("Some Error while writing output")

    # sleeping time for 1 sec
    time.sleep(1)
    end = time.time()

    print(end - start)


def ridge_reg(trainfile, testfile, regularizationfile, outputfile, weightfile, bestparameter):
    start = time.time()

    # Reading Lambdas
    regularization_File = open(regularizationfile, "r")
    line = regularization_File.readline()
    lambdas = list(map(float, (line.split(","))))
    regularization_File.close()
    lambdas.sort()
    # print(lambdas)

    # Reading, Formatting Input Vectors and Output Vectors
    train = pd.read_csv(trainfile).iloc[:, 1:]
    X = train.iloc[:, 0:train.shape[1] - 1].values
    X = np.insert(X, 0, 1, axis=1)
    X_transpose = X.transpose()
    # print(pd.DataFrame(X))

    Y = train.iloc[:, train.shape[1] - 1].values
    # print(pd.DataFrame(Y))

    row = X.shape[0]

    fold_size = X.shape[0] // 10

    minimum_sum = float('inf')
    lam = -1
    for lamda in lambdas:
        curr_sum = 0
        for split in range(10):
            x1 = X[0:split * fold_size, :]
            y1 = Y[0:split * fold_size]

            x2 = X[(split + 1) * fold_size:row, :]
            y2 = Y[(split + 1) * fold_size:row]

            # Training set
            x_tr = np.concatenate((x1, x2), axis=0)
            y_tr = np.concatenate((y1, y2), axis=0)
            # Testing set
            x_te = X[split * fold_size: min((split + 1) * fold_size, row), :]
            y_te = Y[split * fold_size: min((split + 1) * fold_size, row)]
            x_tr_transpose = x_tr.transpose()

            W = np.dot(np.linalg.inv(np.dot(x_tr_transpose, x_tr) + lamda * np.identity(x_tr.shape[1])),
                       np.dot(x_tr_transpose, y_tr))

            y_te_pred = np.dot(x_te, W)

            curr_sum += np.dot((y_te - y_te_pred).transpose(), (y_te - y_te_pred)) / (
                    10 * np.dot(y_te.transpose(), y_te))

        if curr_sum <= minimum_sum:
            minimum_sum = curr_sum
            lam = lamda
    if lam == -1:
        lam = 0.1247

    W = np.dot(np.linalg.inv(np.dot(X_transpose, X) + lam * np.identity(X.shape[1])), np.dot(X_transpose, Y))
    # print(lam)
    test = pd.read_csv(testfile).iloc[:, 1:]
    # print(test.shape)
    X_test = test.iloc[:, 0:test.shape[1]].values
    X_test = np.insert(X_test, 0, 1, axis=1)
    # print(pd.DataFrame(X_test))

    output = np.dot(X_test, W)
    # print(pd.DataFrame(output))

    output_File = open(outputfile, "w")
    for out in output:
        output_File.write(str(out) + "\n")
    output_File.close()

    weight_File = open(weightfile, "w")
    for weight in W:
        weight_File.write(str(weight) + "\n")
    weight_File.close()

    best_Parameter = open(bestparameter, "w")
    best_Parameter.write(str(lam) + "\n")
    best_Parameter.close()

    time.sleep(1)
    end = time.time()
    print(end - start)


def lasso_reg(trainfile, testfile, outputfile):
    # For scaling different matrix, Uses mean and Standard Deviation to scale DataSet
    scale = StandardScaler()

    # Timer Starts
    start = time.time()

    # Reading training file and Using Cross Validation for best lambda
    train = pd.read_csv(trainfile).iloc[:, 1:]
    lam = 0.0012
    train = train.drop(['Birth Weight', 'Race', 'Ethnicity', 'Payment Typology 1', 'Payment Typology 3',
                        'Operating Certificate Number', 'CCS Diagnosis Code', 'APR DRG Code'], axis=1)

    X_train = train.drop('Total Costs', axis=1).values
    Y_train = train['Total Costs'].values

    features = train.drop('Total Costs', axis=1).columns
    # print(features)

    X_train = scale.fit_transform(X_train)

    # print(pd.DataFrame(X_train))
    # print(pd.DataFrame(Y_train))

    test = pd.read_csv(testfile).iloc[:, 1:]
    test = test.drop(['Birth Weight', 'Race', 'Ethnicity', 'Payment Typology 1', 'Payment Typology 3',
                      'Operating Certificate Number', 'CCS Diagnosis Code', 'APR DRG Code'], axis=1)
    X_test = test.values

    X_test = scale.fit_transform(X_test)

    model1 = LassoLars(alpha=lam, eps=2.220446049250313e-16, max_iter=1000).fit(X_train, Y_train)
    feature_Weights = model1.coef_
    # print(pd.DataFrame(feature_Weights))

    old_Features = 0
    for old_feat in feature_Weights:
        if abs(old_feat) <= 5:
            old_Features += 1

    # print(f"{old_Features} have zero coefficient in old Weight Vector")

    # print(pd.DataFrame(X_train))
    # print(pd.DataFrame(Y_train))
    # print(pd.DataFrame(X_test))

    model1.fit(X_train, Y_train)

    # print("Old Feature model Stats:")
    # print(f"Training Accuracy: {model1.score(X_train, Y_train)}")

    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    # print(pd.DataFrame(X_train_train_poly))
    # print(pd.DataFrame(X_train_test_poly))

    poly.fit(X_train, Y_train)
    model2 = LassoLars(alpha=lam, eps=2.220446049250313e-7, max_iter=1000).fit(X_train_poly, Y_train)
    poly_Weights = model2.coef_
    # print(pd.DataFrame(poly_Weights))

    new_feature_weights = 0
    for el in poly_Weights:
        if abs(el) <= 8:
            new_feature_weights += 1;

    # print(f"{new_feature_weights} have zero coefficient in Weight Vector")

    poly_features = poly.get_feature_names(features)

    # print("Polynomial Model Stats:")
    # print(pd.DataFrame(poly_features))
    # print(f"Training Accuracy: {model2.score(X_train_poly, Y_train)}")

    X_test_poly = poly.fit_transform(X_test)
    Y_test = model2.predict(X_test_poly)

    output_C = open(outputfile, "w")
    for output in Y_test:
        output_C.write(str(output) + "\n")
    output_C.close()

    end = time.time()
    if end - start < 60:
        print(f"{end - start} seconds")
    else:
        print(f"{(end - start) / 60} minutes and {(end - start) % 60} seconds")


# print("Linear Regression")
# print()
# lin_reg("data/train.csv", "data/test.csv", "results/a/output_a.txt", "results/a/weight_a.txt")
# print("Ridge Regression")
# print()
# ridge_reg("data/train.csv", "data/test.csv", "regularization.txt", "results/b/output_b.txt", "results/b/weight_b.txt",
#           "results/b/bestparameter.txt")
# print("Lasso Regression")
# print()
# lasso_reg("data/train_large.csv", "data/test.csv", "results/c/output_c.txt")

n = len(sys.argv)

if n < 5 or n > 8:
    print("Length of Arguments is not matching")
else:
    mode = sys.argv[1]
    if mode == 'a':
        if n == 6:
            try:
                lin_reg(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
            except:
                print("Some Error Occurred during input a !!")
        else:
            print("Error wrong input format a !!")
    elif mode == 'b':
        if n == 8:
            try:
                ridge_reg(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
            except:
                print("Error wrong input format b!!")
        else:
            print("Error wrong input format b!!")
    elif mode == 'c':
        if n == 5:
            try:
                lasso_reg(sys.argv[2], sys.argv[3], sys.argv[4])
            except:
                print("Error wrong input format c !!")

        else:
            print("Error wrong input format c !!")
    else:
        print("No such Mode exists")

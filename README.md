# SPARCS-Hospital-Price-Predictor

### Problem Statement- 

Given Training dataset of [SPARCS Hospital](https://www.cse.iitd.ac.in/~cs5170401/Assignment_1.zip) with 1.6 Million rows, predict the total price of hospital aid with the 
help of 30 features ranging from the type of disease, to ZIP code of hospital, Risk Factor, type of procedure used etc. If you want you can check the original dataset and 
read about individual features on [this](https://healthdata.gov/State/Hospital-Inpatient-Discharges-SPARCS-De-Identified/nff8-2va3).

Now, the problem has been solved using only numpy and scipy libraries of Python.

### The problem has been divided into 3 parts: [source code: linear.py]

### a. 

Solve the problem by using L2 norm of difference in predicted value and actual value as our loss function. Also using a parameter b to absord the bias in the dataset.

![alt text](https://s3.ap-south-1.amazonaws.com/afteracademy-server-uploads/l2-loss-function.png)

### b. 

Solve the problem using ridge regression only. To choose optimal regularization hyperparameter lambda use 10 fold [cross-validation](https://www.cs.cmu.edu/~schneide/tut5/node42.html), use lambda values ranging from 0 to 10000. It was found
that when symmetric dataset was given then a smaller value of lambda was optimal (0.003) and if unsymmetrical data set was given then a larger value of lambda was found to be 
optimal(100). Here, optimal lambda for regularization reduces the loss function the most. 
  
![alt text](https://miro.medium.com/max/1126/1*7WR8ORB7cHNOJYZRBU5a1Q.png)

### c.

Apply feature creation and selection to get the best possible prediction on unseen data by creating additional non-linear features. For this part we use train-large.csv file for
training our model. As the total number of features we want must be less than 300 so we need to precisely select features that are most important. Here, we used feature selection
techniques like studying nature of the feature and seeing correlation of feature with outcome, and plotting graphs (in graphs folder) (features like some codes were not so important
and ZIP code was also not important as facility_id was already given in the features). We were able to reduce useful features from
30 to 22. Then we applied Lasso regression which will select a small number of features(<300). Now, we have to select an optimal regularization penalty lambda for lasso
regression using cross-validation on the data. All of this have benn done in feature_selection.py and the important features are then hardcoded in linear.py for final prediction.
Which got close to 78% accuracy on unseen dataset.


   

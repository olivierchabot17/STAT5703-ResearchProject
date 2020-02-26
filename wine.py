# If running the .py script from RStudio, run the following three lines from
# an R console first. 
# install.packages("reticulate")
# library(reticulate)
# py_install("pandas")
# py_install("sklearn")
# py_install("numpy")
# py_install("keras")

# The following is taken from a tutorial on Data Camp. 
# Import pandas 
import pandas as pd

# Read in datasets
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep = ';')
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep = ';')

# Preview data
white.head()
red.head()

# Look at structure of data 
print(white.info())
print(red.info())

# Similar to the "summary" function in R 
white.describe()
red.describe()

# Check for null values (there are none)
pd.isnull(white)
pd.isnull(red)

# Data pre-processing 
# Combine the two datasets, adding a column for their labels
red['type'] = 1
white['type'] = 0
wines = red.append(white, ignore_index = True)

# Split into test and train 
from sklearn.model_selection import train_test_split

# Specify the data
# Note: ix is deprecated as of Jan 2020; has been changed to iloc. 
# Note: 11 is not included here. Operator is non-exclusive on the right
# Also note: For some reason, they're throwing away the "quality" variable. Not sure why. 
X = wines.iloc[:, 0:11]

# Grab the "type" variable and put it in a numpy array
import numpy as np
y = np.array(wines.type)

# Split the data up in train and test sets
# Random state is the seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# The two datasets are "far apart", should standardize them. Note that we are doing this 
# AFTER we've separated into train and test. (We should not use additional info from the
# test set to inform our standardization of the train set)
# Import `StandardScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import StandardScaler

# Compute the mean and sd to be later used for standardizing
scaler = StandardScaler().fit(X_train)

# Standardize the train and test sets
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Going to do a binary classification 
# Will use multi-layer (3 layers) perceptron. We don't have much data, so 3 is appropriate.
# Activation functions: input: relu, hidden: relu, output: sigmoid 
# Layers are all "Dense" layers 
from keras.models import Sequential
from keras.layers import Dense

# Use Keras' "Sequential" model; initialize the model
model = Sequential()

# Add the first layer (an input layer) 
# The input it's taking is 11 elements 
# The first argument, 12, is the number of hidden units 
# Can change this number; allowing more hidden units = able to learn more complex things, 
# But setting it too high can result in overfitting. 
model.add(Dense(12, activation = 'relu', input_shape = (11, )))

# Add one hidden layer 
model.add(Dense(8, activation = 'relu'))

# Add an output layer 
model.add(Dense(1, activation = 'sigmoid'))

# Examine the results: 
# Output shape: each observation has one value for the output 
# (A number between 0 and 1, representing probability)
model.output_shape

# Model summary (gives the number of layers, and the output shape for each layer)
model.summary()

# Compile the model 
# Use the loss function "binary_crossentropy"; since we're doing binary classification
# Use the ADAM optimization algorithm 
# Turn "accuracy" on to keep track of this metric
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Fit the model 
# Use a batch_size of 1 (the number of observations included in each sample)
# The weights get updated after each sample. 
# Thus, a batch size of 1 means every single observation gets to update the weights on its own,
# once per epoch.
# It also means batch_size = 1 will be EXTREMELY sensitive to any outliers. 
# Larger batch sizes will not be as sensitive to outliers, but also may not be as accurate.
# Larger batch sizes = less computationally expensive.
# using 20 epochs (iterations): number of times the entire training data gets exposed to the network
# Setting verbose = 1 will display progress bar during running
model.fit(X_train, y_train, epochs = 20, batch_size = 1, verbose = 1)

# Use model to predict the labels of the test set
y_pred = model.predict(X_test)

# Evaluate model performance

# Use the "evaluate" function; provides the value of the loss function, and proportion correct
score = model.evaluate(X_test, y_test, verbose = 1)
score

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

# Need to round the predictions to 1 or 0 in order to look at the performance metrics
y_pred_classes = np.round(y_pred, decimals = 0)

# Confusion matrix: diagonals show correct predictions by class, 
# off-diagonals show how many incorrect
confusion_matrix(y_test, y_pred_classes)

# Precision: a measure of the classifier's exactness; higher means more accurate
precision_score(y_test, y_pred_classes)

# Recall: a measure of the classifier's completeness; higher means more cases covered
recall_score(y_test, y_pred_classes)

# F1 score: weighted average of precision and recall 
f1_score(y_test, y_pred_classes)

# Cohen's kappa: accuracy normalized by class imbalance
cohen_kappa_score(y_test, y_pred_classes)

# What if we try to predict wine quality rather than type of wine? 
# This will be a regression problem rather than a binary classification problem. 
# We are assuming quality is a continuous variable for our purposes.
# Note: we could have also treated this as a multi-class classification problem if we wanted to. 

# Data pre-processing 
# This time, our labels are in the "quality" variable 
y = wines.quality

# And our data is all of the columns in the wines dataframe, minus the quality variable 
X = wines.drop('quality', axis = 1) 

# Standardize the data 
# Note: this time, we'll be using k-fold CV, so we are standardizing beforehand
# as opposed to last time when we split into test/train first, then standardized. 
# Note: doing it all in one line of code this time instead of two. 
X = StandardScaler().fit_transform(X)

# Create model 
# Initialize the model
# Multi-layer perceptron (2 layers this time; no hidden layer)
model = Sequential()

# Add input layer 
# Note: we are using input_dim this time instead of input_shape; either should have worked
model.add(Dense(64, input_dim = 12, activation = 'relu'))
    
# Add output layer 
# Note: Doesn't include an activation function (typical when using regression to predict
# a single continuous variable) 
model.add(Dense(1))

# Compile and fit the model, use k-fold CV  
from sklearn.model_selection import StratifiedKFold

# Set seed for reproducibility 
seed = 7
np.random.seed(seed)

# Use MSE as the loss function (typical for regression)
# Use rsmprop as the optimization algorithm (a popular choice)
# Also record the Mean Absolute Error as a performance metric
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
for train, test in kfold.split(X, y):
    #model = Sequential()
    #model.add(Dense(64, input_dim = 12, activation = 'relu'))
    #model.add(Dense(1))
    model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mae'])
    model.fit(X[train, ], y[train], epochs = 10, verbose = 1)

# Use model to predict quality of wine 
y_pred = model.predict(X[test, ])

# Save the two performance metrics from the model run into appropriately named variables
mse_value, mae_value = model.evaluate(X[test], y[test], verbose=0)
mse_value
mae_value 

# Can also compute R^2 value 
from sklearn.metrics import r2_score
r2_score(y[test], y_pred)

# Not all that great, can try parameter tuning. 

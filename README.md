# tensorforest

Google's AI technology TensorFlow goes much beyond deep learning. For classification / regression problems with smaller data size, deep learning techniques aren't suitable since deep learning models have thousands of parameters requiring large datasets. A popular model choice for smaller datasets is Random Forest Regression / Classification.

In this project we explore the power of TensorFlow to fit a random forest regression model on the [House Price Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) taken from kaggle. Thge dataset is small (1'460 samples) and thus random forest is the perfect candidate.

## 1. TensorFlow

Imports
```python
import tensorflow as tf
```

Building model params for a random forest with 50 trees, max nodes = 20, number of X variables / features = 79 and number of y variables / classes = 1
```python
params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
  num_classes=1, num_features=79, 
  num_trees=50, max_nodes=20, regression=True).fill()
```

Training the model on (x, y) where is an n x 79 and y is an n x 1 numpy arrays
```python
regressor = tf.contrib.learn.TensorForestEstimator(params)
regressor.fit(x=x, y=y, steps=100)
```

Finally predicting y from new samples of x
```python
y_new = regressor.predict(x_new)
```


## 2. We train a random forest on the house price data from kaggle
### 2.1. Model
We train a random forest regression model. A random forest regression model works similarly to a random forest classification model, however, in the leaf nodes instead of assigning class labels, the model fits a regression line.

### 2.2. Input Data

The input data has 1'460 samples with 79 features containing of continuous as well as categorical features (number of X variables) representing an attribute of a house. The target variable (y) is the sale price of a house which is continuous.

We roughly split the input data as 80% train and 20% validation. Train data has 1'168 samples.

### 2.3. Train

#### 2.3.1. 5-fold cross validation training/test data

<br>

The 5-fols x-validation training works like this:

* The 1'168 data samples are randomly split in roughly 5 equal parts S[i] indexed by i=1, 2, .., 5
* For each i, S[i] is taken as the test set and the model is trained on the remaining 4 sample sets
* Performance on all the test sets S[i] are averaged to get a final performance metric
* The performance metric we use is R-squared=1-RSS/TSS
* This performance metric is compared for different values of the tuning parameter

<br>

#### 2.3.2. Impact of number of trees

<br>

![Impact of Tree number](https://github.com/indiquant/tensorforest/blob/master/examples/images/num_trees.png)

<br>

#### 2.3.3. Choosing max nodes

<br>

We tried a number of values for our tuning parameter MAX NODES. As we keep on increasing the tuning parameter MAX NODES we see a sharp improvement in performance initially, and then the performance gain flattens out. The out of sample test R2 is 0.80 for 50 MAX NODES, 0.82 for 100 MAX NODES and 0.85 for 500 MAX NODES. 

We choose 500 MAX NODES and 100 trees for our final model.

<br>

![Impact of Max nodes](https://github.com/indiquant/tensorforest/blob/master/examples/images/max_nodes.png)

<br>

#### 2.3.4. Final model performance on validation set

<br>

![Final Model Performance](https://github.com/indiquant/tensorforest/blob/master/examples/images/prediction_logscale.png)

Validation R^2 = 0.876
log scale RMSE = 0.135

<br>

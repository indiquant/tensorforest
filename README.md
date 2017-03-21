# tensorforest
using google's AI stack Tensor Flow to implement random forest

## 1. TensorFlow


## 2. We train a random forest on the house price data from kaggle
### 2.1. Model
We train a random forest regression model

### 2.2. Input Data

The input data has 1'460 samples with 79 features containing of continuous as well as categorical features (number of X variables) representing an attribute of a house. The target variable (y) is the sale price of a house which is continuous.

We roughly split the input data as 80% train and 20% validation. Train data has 1'168 samples.

### 2.3. Train

#### 2.3.1 5-fold cross validation training/test data

<br>

The 5-fols x-validation training works like this:

* The 1'168 data samples are randomly split in roughly 5 equal parts S[i] indexed by i=1, 2, .., 5
* For each i, S[i] is taken as the test set and the model is trained on the remaining 4 sample sets
* Performance on all the test sets S[i] are averaged to get a final performance metric
* The performance metric we use is R-squared=1-RSS/TSS
* This performance metric is compared for different values of the tuning parameter

<br>

#### 2.3.2 Impact of number of trees

<br>

![Alt Text](https://github.com/indiquant/tensorforest/blob/master/examples/images/num_trees.png)

<br>

#### 2.3.3 Choosing max nodes

<br>

We tried a number of values for our tuning parameter MAX NODES. As we keep on increasing the tuning parameter MAX NODES we see a sharp improvement in performance initially, and then the performance gain flattens out. The out of sample test R2 is 0.80 for 50 MAX NODES, 0.82 for 100 MAX NODES and 0.85 for 500 MAX NODES.

Even though 500 MAX NODES give the best performance, however 50 or 100 MAX NODES will result in a much stabler model without compromising the performance gain much. We choose 50 as the value of MAX NODES for our tuning param.

<br>

![Alt Text](https://github.com/indiquant/tensorforest/blob/master/examples/images/max_nodes.png)

<br>

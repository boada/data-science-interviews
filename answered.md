# Theoretical interview questions

* The list of questions is based on this post: https://hackernoon.com/160-data-science-interview-questions-415s3y2a
* Legend: 👶 easy ‍⭐️ medium 🚀 expert
* Do you know how to answer questions without answers? Please create a PR
* See an error? Please create a PR with fix

## Supervised machine learning

**What is supervised machine learning? 👶**

A case when we have both features (the matrix X) and the labels (the vector y). 

In more detail: Supervised learning is where you are trying to predict the values of some response variable for various values of the predictor variables (the *features*). For supervised learning to be possible, you need to be able to train your model on data for which you know the values of the response variable already. After doing that, you use your model to predict the values of the response variable for values of the predictors that your model hasn't seen yet. 

## Linear regression

**What is regression? Which models can you use to solve a regression problem? 👶**

Regression is a part of supervised ML. Regression models predict a real number

**What is linear regression? When do we use it? 👶**

Linear regression is a model that assumes a linear relationship between the input variables (X) and the single output variable (y).

With a simple equation:

```
y = B0 + B1*x1 + ... + Bn * xN
```

B is regression coefficients, x values are the independent (explanatory) variables  and y is dependent variable.

The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression.

Simple linear regression:

```
y = B0 + B1*x1
```

Multiple linear regression:

```
y = B0 + B1*x1 + ... + Bn * xN
```

Linear regression assumes four things: 

1. That the response variable is a linear combination of the predictors
2. That the distribution of the errors is normal
3. The errors are homoskedastic w.r.t. both the response variable and the predictors
4. The errors are statistically independent of each other


<br/>

**What’s the normal distribution? Why do we care about it? 👶**

The normal distribution is a continuous distribution given by 

$$\ \frac 1{\sigma\sqrt{2\pi}}{e^{-\frac{{(x-\mu)^2}}{2{\sigma^2}}}} $$

Many phenomena in nature, such as human heights, follow this distribution. The Central Limit Theorem says that for any parameter you are trying to estimate by taking samples, as the number of samples approaches infinity, the distribution will approach the normal distribution, with mean identical to the true mean value of the parameter. 

<br/>

**How do we check if a variable follows the normal distribution? ‍⭐️**

A quick-and-dirty test is a q-q plot. Figure out what the quantiles of the variable should be if it follows a normal distribution. Then generate a quantile-by-quantile plot, plotting these quantiles against the observed ones. If the plot is similar to the line y = x then that is a good indicator that the variable follows the normal distribution. 

A more rigorous test is the Shapiro-Wilk test. With a threshold of 0.05, if the p-value is less than 0.05 then we can conclude that the sample does not come from a normal distribution; otherwise it does. 

<br/>

**What if we want to build a model for predicting prices? Are prices distributed normally? Do we need to do any pre-processing for prices? ‍⭐️**

Answer here

<br/>

**What methods for solving linear regression do you know? ‍⭐️**

Answer here

<br/>

**What is gradient descent? How does it work? ‍⭐️**

When we're training a model, we have some loss function that we want to minimize. When we make a mistake, we want to then make each parameter better so that, if presented with the same training example again, we'd do better. So we take the partial derivative of the loss function w.r.t. each parameter, consider the vector of those partial derivatives (the gradient), then simultaneously update each parameter in the steepest negative direction of the gradient. How much do you update these parameters? ("step size") That is specified by the learning rate parameter. 

With a momentum-based optimizer, you check the steepness of the gradient and let the learning rate depend on that. If it's really steep you take a big step; if shallow then a small step. 

<br/>

**What is the normal equation? ‍⭐️**

The normal equation is an analytic approach to solving linear regression. It is computationally expensive because you have to invert a very large matrix.

Normal equations are obtained by setting equal to zero the partial derivatives of the sum of squared errors (least squares); normal equations allow one to estimate the parameters of a multiple linear regression.

<br/>

**What is SGD  —  stochastic gradient descent? What’s the difference with the usual gradient descent? ‍⭐️**

In stochastic gradient descent, you pick random examples and use that to update your weights. Stochastic GD can be noisy because you could step in the wrong direction; you're only taking a few examples at a time to update your weights. In batch gradient descent you update with all the data you have. You go linearly through the dataset and train on one example at a time. Mini-batch is a middle ground

<br/>

**Which metrics for evaluating regression models do you know? 👶**

The mean square error, root mean square error, median absolute deviation, normalized median absolute deviation, etc.

Residual plots are a good way of seeing whether or not your model is homoskedastic. 

<br/>

**What are MSE and RMSE? 👶**

MSE is the mean squared error-- the mean error of the squared distance between the predicted value of the response variable and the observed value. RMSE is Root Mean Squared Error-- the square root of MSE.

<br/>


## Validation

**What is overfitting? 👶**

Overfitting is when a model is tuned to perform very well on the training data set but cannot generalize to the test data set. It has low bias and high variance.
<br/>

**How to validate your models? 👶**

The first step is to split the full data set into a training or testing subset. This ensures that the model is only trained on a (representative) portion of the data, and provides us with data with which to compare the predictions of our model.
<br/>

**Why do we need to split our data into three parts: train, validation, and test? 👶**

* Train -- This is the data set used to train the model.
* Validation -- This portion of the data is used to initially evaluate the model's performance and to perform any hyperparameter tuning or other optimizations.
* Test -- This is the final hold out set of data that the model has never seen before.

It is important to have *three* parts of the data to ensure that the model is the as flexible as it can be. The validation portion allows us to tune the model, but still have a portion of the data that is useful for evaluating the final performance.
<br/>

**Can you explain how cross-validation works? 👶**

Cross-validation is the process to separate your total training set into two subsets: training and validation set, and evaluate your model to choose the hyperparameters. But you do this process iteratively, selecting different training and validation sets, in order to reduce the bias that you would have by selecting only one validation set.
<br/>

**What is K-fold cross-validation? 👶**

K-fold cross validation is when we "fold" the training data K times, generating K training-validation sets.
<br/>

**How do we choose K in K-fold cross-validation? What’s your favorite K? 👶**

It largely depends on how many training examples we have. If there are only 10 training examples, then *k=10* wouldn't make much sense, as there is only a single training example in each case.

My favorite is *k = 5* because that is roughly an 80-20 split between training and validation.
<br/>


## Classification

**What is classification? Which models would you use to solve a classification problem? 👶**

Classification is labeling a set of observations into two or more categories. You can use many of the same types of models for classification that you could use for regression.
<br/>

**What is logistic regression? When do we need to use it? 👶**

logistic regression is a linear-type model where the predicted value is either 0 or 1. See this: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

Logistic regression assumes that the predictors are linearly independent, and that for each response class and each data point, the log odds (log of the odds) that that data point falls in that class are a linear function of the predictor values for that data point. In the binary case, the odds are defined as: p(x) / ( 1 - p(x) ). 

The loss function for logistic regression is:
J(y-hat) = { -log(1 - y-hat), for y = 0
	            -log(y-hat), for y = 1

Note that:
1. If y = 0 and y-hat is close to 0, J(y-hat) is small. 
2. If y = 0 and y-hat is close to 1, J(y-hat) is large. 
3. If y = 1 and y-hat is close to 0, J(y-hat) is large. 
4. If y = 1 and y-hat is close to 1, J(y-hat) is small. 

<br/>

**Is logistic regression a linear model? Why? 👶**

Yes. [This](https://www.quora.com/Why-is-logistic-regression-considered-a-linear-model?share=1) seems like a good answer.

From that link:

> Logistic regression is considered a generalized linear model because the outcome always depends on the sum of the inputs and parameters. Or in other words, the output cannot depend on the product (or quotient, etc.) of its parameters!

Here's a bit more: Logistic regression produces a linear decision boundary because the additivity of the terms: Our outcome z depends on the additivity of the parameters, e.g., :
```
z = w_1 * x_1 + w_2 * x_2
```
There's no interaction between the parameter weights, nothing like w_1*x_1 * w_2* x_2 or so, which would make our model non-linear!


It's not the same kind of linear model as you have in simple linear regression, though. Simple linear models assume that the response variable is a linear combination of the predictors. Logistic regression assumes that the *log of the odds of the response variable taking the positive class* is a linear combination of the predictors. (That is, log(p/(1-p)), where p is the probability that the value of the response variable is 1.) 


<br/>

**What is the sigmoid function? What does it do? 👶**

A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve. A common example of a sigmoid function is the logistic function, defined by the formula

$$\ S(x) = \frac 1{1}{1 + e^{-x}} $$


The logistic function takes values between 0 and 1. In the context of logistic regression, this allows us to model the probability of a data point's value for the response variable being 1, assigning that probability a value between 0 and 1, as needed. 

<br/>

**How do we evaluate classification models? 👶**

We compare the predicted classes to the true classes. In general, the more observations where we correctly predict the class the better the model is.
<br/>

**What is accuracy? 👶**

Accuracy is a metric for evaluating classification models. It is calculated by dividing the number of correct predictions by the number of total predictions.

<br/>

**Is accuracy always a good metric? 👶**

Accuracy is not a good performance metric when there is imbalance in the dataset. For example, in binary classification with 95% of A class and 5% of B class, prediction accuracy can be 95%. In case of imbalance dataset, we need to choose Precision, recall, or F1 Score depending on the problem we are trying to solve.

<br/>

**What is the confusion table? What are the cells in this table? 👶**

Confusion table (or confusion matrix) shows how many True positives (TP), True Negative (TN), False Positive (FP) and False Negative (FN) model has made.

||                |     Actual   |        Actual |
|:---:|   :---:        |     :---:    |:---:          |
||                | Positive (1) | Negative (0)  |
|**Predicted**|   Positive (1) | TP           | FP            |
|**Predicted**|   Negative (0) | FN           | TN            |

* True Positives (TP): When the actual class of the observation is 1 (True) and the prediction is 1 (True)
* True Negative (TN): When the actual class of the observation is 0 (False) and the prediction is 0 (False)
* False Positive (FP): When the actual class of the observation is 0 (False) and the prediction is 1 (True)
* False Negative (FN): When the actual class of the observation is 1 (True) and the prediction is 0 (False)

Most of the performance metrics for classification models are based on the values of the confusion matrix.

<br/>

**What are precision, recall, and F1-score? 👶**

* Precision (P) and Recall (R) are classification evaluation metrics:
* P = TP / (TP + FP) and R = TP / (TP + FN).
* Where TP is true positives, FP is false positives and FN is false negatives
* In both cases the score of 1 is the best: we get no false positives or false negatives and only true positives.
* F1 is a combination of both precision and recall in one score:
* F1 = 2 * PR / (P + R).
* Max F score is 1 and min is 0, with 1 being the best.

<br/>

**Precision-recall trade-off ‍⭐️**

For a given model, we can either increase the precision or the recall at the expense of the other. If we want to accurately predict a few high-risk situations we would want higher recall, because it might be better to treat everything upfront. If we need to identify more lower-risk situations then precision might be better because the cost of missing some is lower.
<br/>

**What is the ROC curve? When to use it? ‍⭐️**

The Receiver Operating Characteristics (ROC) curve is a performance metric for classifications are various threshold settings. It is plotted as the *false positive rate* vs. *true positive rate* (recall) where the diagonal line represents 50-50 chance.
<br/>

**What is AUC (AU ROC)? When to use it? ‍⭐️**

The area under the curve (AUC) represents a model's ability to discriminate between classes correctly. The higher the AUC the better the model is at correctly predicting classes.

<br/>

**How to interpret the AU ROC score? ‍⭐️**

When the AUC is close to 1 the model is has a good measure of separability between the classes. When the value is close to 0, then the model is predicting exactly the opposite of what we want 1 when should be 0. A value close to 0.5 means that the model has very little power to separate the classes.
<br/>

**What is the PR (precision-recall) curve? ‍⭐️**

Shows the frontier boundary between the precision and recall for a given model. As you increase one the other will decrease, *for the given model.*

<br/>

**What is the area under the PR curve? Is it a useful metric? ‍⭐️I**

Answer here

<br/>

**In which cases AU PR is better than AU ROC? ‍⭐️**

Answer here

<br/>

**What do we do with categorical variables? ‍⭐️**

We either use a model that can handel directly (random forest?) or we transform them into a numerical feature that we can feed into a different model.
<br/>

**Why do we need one-hot encoding? ‍⭐️**

Because encoding categorical variables with a simple linear scale `[1,2,3,..]` can imply different weights to the variables. We use one-hot encoding to convert categorical variables into number values without implicitly assigning them (some times misleading) weights.
<br/>


## Regularization

**What happens to our linear regression model if we have three columns in our data: x, y, z  —  and z is a sum of x and y? ‍⭐️**

Answer here

<br/>

**What happens to our linear regression model if the column z in the data is a sum of columns x and y and some random noise? ‍⭐️**

Answer here

<br/>

**What is regularization? Why do we need it? 👶**

Regularization encompasses a broad range of techniques used to penalize or otherwise avoid overfitting when we are training a model. In the simplest cases, it adds a penalty term that increases as model complexity increases.
<br/>

**Which regularization techniques do you know? ‍⭐️**

For linear models: L1 (lasso), L2 (ridge), elastic net
For single trees: cost-complexity pruning
For ensemble tree methods: sample bagging and feature bagging
For neural networks: early stopping and dropouts

<br/>

**What kind of regularization techniques are applicable to linear models? ‍⭐️**

L1 (lasso), L2 (ridge), elastic net

<br/>

**What does L2 regularization look like in a linear model? ‍⭐️**

L2 regularization penalizes the model with the square of the weights
<br/>

**How do we select the right regularization parameters? 👶**

Plot the cross validation RMSE for each value of the parameter, and choose the value right after the elbow. 

<br/>

**What’s the effect of L2 regularization on the weights of a linear model? ‍⭐️**

It reduces them to small, but non-zero values.
<br/>

**How L1 regularization looks like in a linear model? ‍⭐️**

L1 penalizes the model with the absolute value of the weights.

<br/>

**What’s the difference between L2 and L1 regularization? ‍⭐️**

The penality term. A regression model that uses L1 regularization technique is called *Lasso Regression* and model which uses L2 is called *Ridge Regression*.

* L1 - Lasso Regression (Least Absolute Shrinkage and Selection Operator) adds “absolute value of magnitude” of coefficient as penalty term to the loss function.

* L2 - Ridge regression adds “squared magnitude” of coefficient as penalty term to the loss function.

<br/>

**Can we have both L1 and L2 regularization components in a linear model? ‍⭐️**

Yes, that is called elastic net.
<br/>

**What’s the interpretation of the bias term in linear models? ‍⭐️**

Answer here

<br/>

**How do we interpret weights in linear models? ‍⭐️**

If the variables are normalized, we can interpret weights in linear models like the importance of this variable in the predicted result.

<br/>

**If a weight for one variable is higher than for another  —  can we say that this variable is more important? ‍⭐️**

Yes, but only for the case above.
<br/>

**When do we need to perform feature normalization for linear models? When it’s okay not to do it? ‍⭐️**

Answer here

<br/>


## Feature selection

**What is feature selection? Why do we need it? 👶**

It is selecting only features which significantly contribute to the model. We need to do it because if we are including many low-signal features, then we are just adding noise and confusion to the model and not actually increasing its predictive power (or maybe only slightly).
<br/>

**Is feature selection important for linear models? ‍⭐️**

Yes. For the same reasons as above, including the less-predictive features only adds noise to the model.
<br/>

**Which feature selection techniques do you know? ‍⭐️**

Answer here

<br/>

**Can we use L1 regularization for feature selection? ‍⭐️**

Yes!
<br/>

**Can we use L2 regularization for feature selection? ‍⭐️**

Not really.
<br/>


## Decision trees

**What are decision trees? 👶**

Answer here

<br/>

**How do we train decision trees? ‍⭐️**

We use the CART algorithm. At each decision point in the tree the dataset is split in order to either maximize the purity (classification) or the variance (regression). Repeat this process until the desired depth.

For classification trees, scikit-learn tries to minimize the Gini impurity.

```
G = sum(fi * (1-fi))
```
Summed from 1 to *C*, where *fi* is the frequency of the label *i* at a node and *C* is the number of unique labels (features).

For regression, we are going to use the variance reduction (in each split). The variance is calculated by the Mean Squared Error

```
MSE = 1/N * sum((yi - mu)**2)
```
where *yi* is the label (feature) for an instance, *N* is the total number of instances and *mu* is the mean value of all instances


<br/>

**What are the main parameters of the decision tree model? 👶**

The depth of the tree (the number of split nodes).

<br/>

**How do we handle categorical variables in decision trees? ‍⭐️**

Answer here

<br/>

**What are the benefits of a single decision tree compared to more complex models? ‍⭐️**

Answer here

<br/>

**How can we know which features are more important for the decision tree model? ‍⭐️**

From Toward Data Science:

>Feature importance is calculated as the decrease in node impurity weighted by the probability of reaching that node. The node probability can be calculated by the number of samples that reach the node, divided by the total number of samples. The higher the value the more important the feature.

The importance of each node is given by:

```
nij = wj*Cj - wlefti*Cleftj - wrightj* Crightj
```
where
* nij = the importance of node j
* wj = weighted number of samples reaching node j
* Cj = the impurity value of node j
* leftj = child node from left split on node j
* rightj = child node from right split on node j

See [this](https://towardsdatascience.com/the-mathematics-of-decision-trees-random-forest-and-feature-importance-in-scikit-learn-and-spark-f2861df67e3) Toward Data Science article for more equations.

<br/>


## Random forest

**What is random forest? 👶**

An ensemble method that uses a collection of decision trees to solve the regression or classification problem.
<br/>

**Why do we need randomization in random forest? ‍⭐️**

Training on a random subset of the data helps to prevent overfitting.
<br/>

**What are the main parameters of the random forest model? ‍⭐️**

Number of trees, depth, etc.
<br/>

**How do we select the depth of the trees in random forest? ‍⭐️**

Answer here

<br/>

**How do we know how many trees we need in random forest? ‍⭐️**

Answer here

<br/>

**Is it easy to parallelize training of a random forest model? How can we do it? ‍⭐️**

Answer here

<br/>

**What are the potential problems with many large trees? ‍⭐️**

Answer here

<br/>

**What if instead of finding the best split, we randomly select a few splits and just select the best from them. Will it work? 🚀**

Answer here

<br/>

**What happens when we have correlated features in our data? ‍⭐️**

Answer here

<br/>


## Gradient boosting

**What is gradient boosting trees? ‍⭐️**

Gradient boosting is an ensemble method for improving the performance of classification and regression trees. The regression case proceeds as follows. You start with a single leaf, whose value is the average of the response variable. Then you grow that to a tree of fixed depth (depth is a hyperparameter, controlling the amount of interaction between variables). Instead of trying to predict the response value, here you try to predict the error from the first tree (the single leaf). Then to predict the response value, add the predicted error (scaled by the learning rate) to the prediction from the first tree (the single leaf). Repeat this process, only now trying to predict the residuals of the second iteration. Continue until you have the pre-specified number of trees, or until adding additional trees doesn't decrease the residuals. (Note: in each iteration, you calculate the residuals by adding the predictions from all previous trees.)

<br/>

**What’s the difference between random forest and gradient boosting? ‍⭐️**

Gradient boosted trees (GBT) differ from a random forest because in GBTs we are training one tree at a time, iteratively, one after another, and then aggregating the results. In a random forest, we are using a random subset of the data and training all of the trees in parallel.

Note though: Both gradient boosting and Random Forest are ensemble methods: they involve combining the results of different trees. Also, like Random Forest, stochastic gradient boosting uses bagging. 

<br/>

**Is it possible to parallelize training of a gradient boosting model? How to do it? ‍⭐️**

Answer here

<br/>

**Feature importance in gradient boosting trees  —  what are possible options? ‍⭐️**

I'm only aware of one option: the relative importance of a variable x is (the square root of) the sum of the squared improvements in squared error over all internal nodes for which it was chosen as the splitting variable. 

(Here 'improvements' refers to improvements over the error associated with assigning a constant value over the entire region of the input space that is being split.)

<br/>

**Are there any differences between continuous and discrete variables when it comes to feature importance of gradient boosting models? 🚀**

Answer here

<br/>

**What are the main parameters in the gradient boosting model? ‍⭐️**

The learning rate and the number of trees.
<br/>

**How do you approach tuning parameters in XGBoost or LightGBM? 🚀**

Stochastic search or grid search. 

<br/>

**How do you select the number of trees in the gradient boosting model? ‍⭐️**

It depends on the size of your dataset, but this is a hyperparameter that should be tuned using stochastic search or grid search. 

<br/>



## Parameter tuning

**Which parameter tuning strategies (in general) do you know? ‍⭐️**

Grid and stochastic search with cross validation.
<br/>

**What’s the difference between grid search parameter tuning strategy and random search? When to use one or another? ‍⭐️**

Grid search performs an exhaustive search of all possible parameters available to the model. A random search searches randomly through the parameter space, and then reports the best parameters. In principle you could use a random search to narrow the search space and then a grid search to refine the solution. In practice, I would use random search because it is faster than an exhaustive grid search.

<br/>


## Neural networks

https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464

**What kinds of problems can neural nets solve? 👶**

Both supervised and unsupervised learning problems. Supervised: classification and regression. Unsupervised: automatic target recognition.
<br/>

**How does a usual fully-connected feed-forward neural network work? ‍⭐️**

A feedforward neural network is an artificial neural network in which connections between the nodes do not form a cycle. Fully connected means that in each layer, every node in that layer is connected to every node in the next layer. 

It starts with the nodes in the input layer having certain levels of activation. Then that propagates to the next layer via an activation function for each neuron in the hidden layers-- each hidden neuron's activation level is determined by the activation function, applied to the activation levels of all neurons in the previous layer, plus the weights of their connection to that neuron. This continues from layer to layer until the output layer-- the output is the activation levels of the output layer neurons. 

<br/>

**Why do we need activation functions? 👶**

Activation functions specify how the activation level of a neuron depend on those of the inputs feeding into it (and the weights connecting them). The activation function provides a smooth, differentiable transition in activation levels from layer to layer as input values change; i.e., a small change in input produces a small change in output. We need the transition in activation levels to be smooth and differentiable because the gradient of this function is used when training the network (so it must be well-defined!). 

<br/>

**What are the problems with sigmoid as an activation function? ‍⭐️**

1. Sigmoid saturate and kill gradients: The output of sigmoid saturates (i.e. the curve becomes parallel to x-axis) for a large positive or large negative number. Thus, the gradient at these regions is almost zero. During backpropagation, this local gradient is multiplied with the gradient of this gates’ output. Thus, if the local gradient is very small, it’ll kill the the gradient and the network will not learn. This problem of vanishing gradient is solved by ReLU.


2. Not zero-centered: Sigmoid outputs are not zero-centered, which is undesirable because it can indirectly introduce undesirable zig-zagging dynamics in the gradient updates for the weights.

https://kharshit.github.io/blog/2018/04/20/don%27t-use-sigmoid-neural-nets
<br/>

**What is ReLU? How is it better than sigmoid or tanh? ‍⭐️**

'ReLU' stands for Rectified Linear Unit. It is defined as:

f(x) = max(0,x)

If your learning rate is high and you backpropagate a large signal backwards, the result will be negative and so your neuron's activation will be set to 0, and will continue to be 0 henceforth. You can avoid this by using leaky ReLU.

<br/>

**How we can initialize the weights of a neural network? ‍⭐️**

It's standard to initialize them to be random numbers. The weights might converge to different local minima of the cost function, depending on how they're initialized. So you could try initializing with several different random vectors and seeing if you get the same thing. 

<br/>

**What if we set all the weights of a neural network to 0? ‍⭐️**

Then the network will not provide any outputs. If this is done before training, that makes it impossible to train. 

<br/>


**What regularization techniques for neural nets do you know? ‍⭐️**

Dropout. 

<br/>

**What is dropout? Why is it useful? How does it work? ‍⭐️**

Dropout is a method for reducing overfitting when you are training a neural network. The flexibility of a neural network lies partly in the number of nodes in its hidden layers. Dropout is where, while training, you drop a specified number of randomly selected nodes from a hidden layer. You can do this on multiple hidden layers, and the number can be different for each hidden layer. 

Dropout is useful because it reduces overfitting. Essentially it reduces the complexity of the network. 

<br/>


## Optimization in neural networks

**What is backpropagation? How does it work? Why do we need it? ‍⭐️**

Backpropagation is the algorithm used for training neural networks. It begins by determining, for each data point in the training set, how (in direction and magnitude) that data point would like to nudge the weights and biases of each neuron in the network, to cause the most rapid decreases in the cost function (gradient descent). In an ideal scenario, you would do this for each training example, and then average the desired changes demanded by all of the training examples. However, since that is computationally infeasible, one typically uses stochastic gradient descent instead (another subject)-- randomly divide the data into mini-batches, and compute each step with respect to a mini-batch, going through all mini-batches.

In backpropagation, you start with a single training example. You compare the current output to the desired one, and see what's the difference. Then you decide, for each neuron in the output layer, in which direction its activation level needs to change (up or down) and by how much, for the current output to change to the desired output. Then, for each of these neurons, you examine each neuron in the *previous* layer, and figure out in which direction *its* activation needs to change (and by how much) to produce the effect on the chosen neuron in the output layer. From this, you get an amount that the weight of each connection (of each neuron in the second-to-last layer) to your chosen output neuron, as well as the bias of that neuron, needs to change, in order for your chosen neuron to provide the desired output for that data point. You repeat this for each neuron in the output layer and average the results. But now notice that at this point, you have suggested changes to make to the *second-to-last* layer, just as at the beginning, comparing the output layer with the desired output gave you suggested changes for the output layer. So you repeat the same process, except this time you treat the second-to-last layer as the output layer, and adjust (a) the biases of the neurons in the second-to-last layer and (b) the weights connecting the second-to-last to the third-to-last layer. Then keep repeat the process, "propagating" the suggested changes backward through the layers of the network, until all the layers have been considered. 

<br/>


**When you do forward propagation, are you already adjusting the weights and biases? Or just in backpropagation? ‍⭐️**

Only in backpropagation. 

<br/>


**Which optimization techniques for training neural nets do you know? ‍⭐️**

Mini-batching, ADAM. 

<br/>

**How do we use SGD (stochastic gradient descent) for training a neural net? ‍⭐️**

Answer here

<br/>

**What’s the learning rate? 👶**

In gradient descent, the learning rate is the size of the step that you take along the steepest gradient. In neural networks, the learning rate is the rate at which you update your weights. Each data point demands a certain change to the bias and weights of each neuron. But to avoid overfitting, instead of simply updating the bias and weights by the amount that that data point demands, you instead retard that adjustment by a specified amount-- that amount is the learning rate. 

<br/>

**What happens when the learning rate is too large? Too small? 👶**

If the learning rate is too large, you may miss (overshoot) a local minimum of the cost function. If it's too small, it will take too long / be too computationally expensive to reach a local minimum. 

<br/>

**How does one set the learning rate? ‍⭐️**

This is a hyperparameter that can be tuned. Also, ideally the learning rate should be proportional to the magnitude of the gradient (and so, change as the magnitude of the gradient changes). That way you make finer adjustments as you get closer to the local minimum of the loss function. 

<br/>

**What is Adam? What’s the main difference between Adam and SGD? ‍⭐️**

Answer here

<br/>

**When would you use Adam and when SGD? ‍⭐️**

Answer here

<br/>

**Do we want to have a constant learning rate or is it better to change it throughout training? ‍⭐️**

Ideally, the learning rate should be proportional to the magnitude of the gradient (and so, change as the magnitude of the gradient changes). That way you make finer adjustments as you get closer to the local minimum of the loss function.

<br/>

**What is an RNN? ‍⭐️**

(Much material taken from Ch. 7 of Aggarwal's *Neural Networks and Deep Learning*.)

'RNN' stands for Recurrent Neural Network. These networks are designed to deal with data in which there are sequential dependencies between the inputs. E.g., in a sentence, the order of words matters; there are relationships between the words that would be ignored by representing the sentence as a bag of words (compare 'the cat chased the mouse' with 'the mouse chased the cat'). 

Once we abandon the bag-of-words approach to representing documents, we are faced with the difficulty that the inputs to our network have variable sizes. In an RNN, each position in the sequence of an input has an associated *time-stamp*. An RNN then contains a varying number of layers, and each layer has a single input corresponding to a distinct time-stamp. This allows the inputs to directly interact with down-stream hidden layers depending on their positions in the sequence. 

For example, suppose our task is to predict the next word of a sentence. Given the sentence 'The cat chased the mouse', we might have the following: 
1. First, a network with one hidden layer, whose input is just the word 'the', and whose output is hoped to be 'cat'. 
2. Then, a network with two hidden layers, whose input is 'The cat', and whose output is hoped to be 'chased'. 
3. Then, a network with three hidden layers, whose input is 'The cat chased', and whose output is hoped to be 'the'. 
4. Finally, a network with four hidden layers, whose input is 'The cat chased the', and whose output is (hoped to be) 'mouse'. 

So, the network takes in a sequence of inputs and produces a sequence of outputs, changing its number of layers along the way. Each layer uses the same set of parameters to ensure similar modeling at each time stamp, and therefore the number of parameters is fixed throughout. 

In the above example, there is both an input and output at each timestamp. However, it is possible for either the input or the output to be missing at any particular time-stamp. The choice of missing inputs and outputs depends on the specific application at hand. For example, in a time-series forecasting application, we might need outputs at each time-stamp in order to predict the next value in the time-series. On the other hand, in a sequence-classification application such as sentiment analysis, we might only need a single output label at the end of the sequence corresponding to its class. In general, it is possible for any subset of inputs or outputs to be missing in a particular application.

Once the idea of a varying number of hidden layers is grasped, the key detail about RNNs is that the value of a node in a hidden layer h at a given timestamp t is a function of (a) that node's value at the previous timestamp t-1 (that is, the value of the node at that layer at timestamp t-1), and (b) the value of the node in the previous layer h-1 at timestamp t.

Problems with RNNs:
RNNs are very vulnerable to the exploding and vanishing gradient problems, because they can be very deep (depending on the number of inputs). This is aggravated by the sharing of weights by different "versions" of a hidden node across timestamps. That effectively means that the gradient is being multiplied by the same weight matrix many times during backpropagation. Consider the simple case of a 1x1 matrix, a constant. If the constant is less than 1, the gradient will shrink; if it's greater than 1, the gradient will explode. 

The combination of the vanishing/exploding gradient and the parameter tying across different layers causes the recurrent neural network to behave in an unstable way with gradient-descent step size. That is, the optimal points in the parameter spaces of recurrent networks are often hidden near cliffs or other regions of unpredictable change in the topography of the loss function, which causes the best directions of instantaneous movement to be extremely poor predictors of the best directions of finite movement. Since any practical learning algorithm is required to make finite steps of reasonable sizes to make good progress towards
the optimal solution, this makes training rather hard. There have been several attempts to address this issue, including the introduction of LSTMs. 

<br/>


**What is the motivation for bi-directional RNNs and how do they work?**
In some applications, especially speech recognition, handwriting recognition, and sentence completion, performance can be greatly improved by allowing the RNN to look at elements *later* in the sequence than the one it is trying to predict or classify. In other words, to it can help greatly to consider the context both prior and posterior to a given word or character. Bi-directional RNNs allow for this, by letting the hidden neurons read the input from back to front as well as from front to back as "time" progresses. 

As for the architecture, in a bi-directional RNN, each hidden node has not just one state vector but two: a forward state vector **h_t^f^k** and a backward state vector **h_t^b^k**. (Here as usual the superscript k indicates the depth of the node.) During forward propagation, the forward state vector **h_t^f^k** is updated by **h_(t-1)^f^k** and **h_t^f^(k-1)**, scaled by forward-facing weight matrices, summed, and run through tanh, whereas the backward state vector **h_t^b^k** is updated by **h_(t+1)^b^k** and **h_t^b^(k-1)**, scaled by backward-facing weight matrices, summed, and run through tanh. The forward-facing and backward-facing hidden state vectors do not interact directly at all; they are only connected in that they receive the same input (though in reverse orders) and the output of the network aggregates the forward and backward activation levels of the final layer-- so, during backpropagation, the loss is calculated from both the forward- and backward-looking processes. 

One final point. Let the length of the input sequence be N. Then, during backpropagation, the forward-facing weight matrices are trained on the input sequence from t = N to t = 1, whereas the backward-facing weight matrices are trained on the sequence from t = 1 to t = N. 

<br/>

**When do you use which kind of neural network?**

RNNs are typically used to be used when there's a time or order component. (Another classic task is generating words in a sequence.) But simple RNNs are not used as much anymore, especially in NLP. Now transformers are a significant competitor-- they are essentially CNNs with attention. 

RNNs, LSTMs, and GRUs -- NLP

CNNs -- Image processing, some NLP tasks

Transformers --

Autoencoders --

<br/>


**What is attention?**

In the context of words, attention is a weight matrix for each word, showing how much you need to pay attention to every other word in the sentence. In French we have articles that define gender. For an article you need to pay attention to a lot of the other words in the vicinity. RNNs are sequential, so as the sentence gets longer and longer you can get exploding or vanishing gradients, whereas attention doesn't get distorted in that way since it runs in series rather than in parallel. 

https://distill.pub/2016/augmented-rnns/

https://jalammar.github.io/illustrated-transformer/

<br/>


**How do you decide on the architecture of a neural network?**

Adding more layers and increasing the number of neurons per layer are both ways to make your model more complex, and so in principle they are ways of increasing the model's ability to model complicated problems. In practice, increasing the number of layers of neural networks tends to improve overall test set accuracy, whereas large, shallow networks tend to overfit more — which is one stimulus for using deep neural networks as opposed to shallow neural networks.

A lot of the time people simply try out different combinations of number of layers and neurons per layer. Although overfitting is a concern when adding layers (in addition to when increasing neurons per layer), dropout can be used to counteract that. And for certain problems having many layers makes a huge difference. E.g., facial recognition. A different concern that can affect architectural choices is computational expense. But with transfer learning it's less of an issue. 

In all then: the common practice is simply to add more layers in a sequential manner and see how much better the performance gets. 20-25% dropout is the norm. 


https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98

https://towardsdatascience.com/neural-network-architectures-156e5bad51ba

https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.79433&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false

https://arxiv.org/pdf/1311.2901.pdf

<br/>


**How do we decide when to stop training a neural net? 👶**

Answer here

<br/>

**What is model checkpointing? ‍⭐️**

Answer here

<br/>

**Can you tell us how you approach the model training process? ‍⭐️**

Answer here

<br/>


## Neural networks for computer vision

**How we can use neural nets for computer vision? ‍⭐️**

Answer here

<br/>

**What’s a convolutional layer? ‍⭐️**

Answer here

<br/>

**Why do we actually need convolutions? Can’t we use fully-connected layers for that? ‍⭐️**

Answer here

<br/>

**What’s pooling in CNN? Why do we need it? ‍⭐️**

Answer here

<br/>

**How does max pooling work? Are there other pooling techniques? ‍⭐️**

Answer here

<br/>

**Are CNNs resistant to rotations? What happens to the predictions of a CNN if an image is rotated? 🚀**

Answer here

<br/>

**What are augmentations? Why do we need them? 👶What kind of augmentations do you know? 👶How to choose which augmentations to use? ‍⭐️**

Answer here

<br/>

**What kind of CNN architectures for classification do you know? 🚀**

Answer here

<br/>


**What is the vanishing gradient problem? ‍⭐️**

<br/>


**What is an LSTM? ‍⭐️**

https://youtu.be/LHXXI4-IEns

https://youtu.be/8HyCNIVRbSU

(Closely inspired by Aggarwal's *Neural Networks and Deep Learning*, Sec. 7.5)

An LSTM (Long Short-Term Memory network) is a particular type of RNN, which is designed to solve the vanishing/exploding gradient problem. One way of viewing
this problem is that a neural network that uses only multiplicative updates is good only at learning over short sequences, and is therefore inherently endowed with good short-term memory but poor long-term memory. In response, the LSTM introduces a kind of long-term memory. 

In LSTMs, we change the way that hidden states are propagated. In addition to the hidden state vector **h_t^k** (whose dimension p reflects the way the embedding was done), there is another vector **c_t^k** called the *cell state*. (Here I use **boldface** for vectors) The cell state is a kind of long-term memory that retains at least a part of the information in earlier states by using a combination of partial “forgetting” and “increment” operations on the previous cell states. 

These operations are accomplished via several extra elements:
1. An input gate **i**
2. A forget gate **f**
3. An output gate **o**
4. A new c-state **c** (distinct from **c_t^k**)

When we update at a given timestamp t the state vector of a hidden neuron **h_(t-1)^k**, the motto is: "selectively forget and add to long-term memory, and then selectively leak long-term memory to the hidden state **h_t^k**". More precisely:

First, as in all RNNs, updates to a hidden node **h_t^k** are partly determined by the values of the previous (less deep) node **h_t^(k-1)** and that node's value at the previous timestamp, **h_(t-1)^k**, scaled by the weights of the connections between those nodes and **h_t^k**. This gives us a vector of length 4p. We then apply the sigmoid function to the first 3p-many elements of that vector, and tanh to the final p-many elements. This defines the vectors **i**, **f**, **o** and **c**, respectively. We then update the cell state and hidden state vector as follows:


**c_t^k** = **f** x **c_(t-1)^k** + **i** x **c**, where 'x' denotes elementwise vector multiplication. ("selectively forget and add to long-term memory")
Observe: forgetting happens when we multiply **f** by **c_(t-1)^k**; adding to long-term memory happens when we multiply **i** by **c**. 

Then we update **h_t^k**:
**h_t^k** = **o** x tanh(**c_t^k**)   (selectively leak long-term memory to the hidden state **h_t^k**)
Here "leaking long-term memory to **h_t^k**" happens because **h_t^k** is defined partly in terms of **c_t^k**, which represents long term memory. Of course, as usual, **h_t^k** is also influenced more directly by **h_(t-1)^k** and **h_t^(k-1)**, via the output term **o**. 

As Aggarwal explains it, the reason that LSTMs avoid the vanishing gradient problems stems from the fact that by the definition of **c_t^k**, its partial derivative w.r.t. **c_(t-1)^k** is **f**. This means that backward gradient flows for **c_t^k** are multiplied by the value of **f**. If **f** is initialized with a high bias term, gradient flows will decay relatively slowly (getting multiplied by something that's close to 1). Moreover, **f** can take different values at different timestamps, which also mitigates the vanishing gradient problem. As Aggarwal puts it, "the long-term cell states function as gradient super-highways, which leak into hidden states" (via the definition of **h_t^k** partly in terms of **c_t^k**).

<br/>

**What is an autoencoder? ‍⭐️**

You take a high information content Thing, like an image, and run it through a bottleneck (a layer that has very few neurons) to reduce time and space requirements. It is like a non-linear dimensionality reduction method. 

Decreasing numbers of neurons, and then re-increasing number of neurons, which is a decoder. 

Google uses it when you download images from Google. 

Variational auto encoder: in a regular encoder, in the middle there's a vector that doesn't change. In a variational auto-encoder, there's a distribution rather than a vector. You can sample the distribution and create things that didn't exist before. The bottleneck is not a vector but rather a distributon. 

At the bottleneck, each neuron has a set of values: the weights and biases, represented as a vector. But in variational autoencoders, the neurons in the squished layer instead vary within a given distribution(s). 

https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776

<br/>


**What is transfer learning? How does it work? ‍⭐️**

It used to be that you just train your network from scratch. FOr really complex problems you'd train it for weeks or months. But, you can actually take a network that's already trained on one dataset, and apply it to a different problem. That is transfer learning. All you have to do is fine tune it-- e.g., take the pre-trained network and train the last layer. If you don't have a million images, you just apply transfer learning, and because your network is already pre-trained it's already pretty far along. 

You can take a network trained on household images and detect galaxies.  ResNet is the state of the art for image classification. 

Why just the last layer? If we have a really deep network, could we take more off? It's a time/cost consideration. 

And, in what kinds of situations would transfer learning not be a great idea? Very rarely do people train image classification models from scratch. People initially thought transfer learning couldn't be used for NLP, but now it's routine! 

All that being said, for simple problems training your own models is ok. If you're an AI researcher you might need to do your own thing. If your domain is really different from what the network was trained in; e.g., usually you wouldn't use a network trained on images to do NLP. Though there are a few stories of this working well. 

<br/>

**Why is there so little transfer learning associated with RNNs? ‍⭐️**
For contingent historical reasons. As they were doing more RNNs, they came out with transformer networks, which got standardized. It's straightforwardo to build your own LSTM using pytorch. For language, there's basically nothing. For time series analysis, there are some people trying to do transfer learning. People did have trained word embeddings beforehand, such as GLOVE and word2vec, which can be put in as the first layer of your LSTM. 

<br/>


**What is object detection? Do you know any architectures for that? 🚀**

Answer here

<br/>

**What is object segmentation? Do you know any architectures for that? 🚀**

Answer here

<br/>


## Text classification

**How can we use machine learning for text classification? ‍⭐️**

There are several different kinds of task: topic classification (spam or not; appropriate or not, etc.), sentiment analysis, language detection. 

Some of the most popular machine learning algorithms for creating text classification models include logistic regression, gradient boosted trees, the naive bayes family of algorithms, support vector machines, and deep learning.
 
https://monkeylearn.com/text-classification/

https://developers.google.com/machine-learning/guides/text-classification

https://developers.google.com/machine-learning/guides/text-classification/step-2-5

http://uc-r.github.io/creating-text-features#ngrams

https://machinelearningmastery.com/what-are-word-embeddings/ 

https://towardsdatascience.com/word-bags-vs-word-sequences-for-text-classification-e0222c21d2ec 

https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089

<br/>


**What is tokenizing? ‍⭐️**

Tokenizing is when you break up a text, usually into its constituent words and store them separately. There are tricky questions sometimes about how to break things up. E.g., should we break "aren't" into "aren" and "t"? That produces garbage. But if you break it into "are" and "not", you are inserting characters that weren't there before. 

<br/>


**What is bag of words? How we can use it for text classification? ‍⭐️**

Bag of words is a strategy for representing texts (*documents*) as numeric vectors, so that computers can investigate and operate on them. With bag of words, you start with a predefined, ordered dictionary of words. Then any document can be represented as a vector of the same length as the dictionary, where the value for each place in the vector is the count, in that document, of the word occurring in that place in the dictionary. For example, suppose we have defined our dictionary to have the following words: [This, is, the, not, awesome, bad, basketball], and suppose we want to vectorize the document consisting of the single sentence “This is awesome”. We would represent the document by the following vector: [1, 1, 0, 0, 1, 0, 0]. Given a representation of documents as numeric vectors, we can then calculate various things, such as cosine similarity of two vectors, to give a measure of how similar two documents are. For classification tasks, we can tag these vectors with categories in a response variable, and train a classification model, such as logistic regression or tree-based classifiers, on the result.

In a variation on the above, you can use N-grams instead of individual words-- see below. 

<br/>

**What are the advantages and disadvantages of bag of words? ‍⭐️**

One obvious disadvantage is that bag of words (in its simplest implementation-- see below) ignores grammar entirely. In the above example, "This is awesome" is the same as "is awesome this"; they are both represented by the vector [1, 1, 0, 0, 1, 0, 0]. Another disadvantage is that bag of words more or less ignores associations between words, such as between 'miserable' and 'distraught'. A third disadvantage is that *stopwords* (words such as 'is' and 'the') occur very frequently in most documents. So unless the stopwords are removed, two objectively dissimilar documents might easily be counted as similar because of containing similar proportions of stopwords. On the other hand, when you remove stopwords entirely, you are assuming they have essentially *no* effect on the identity of a document, which might be overkill. 

All that being said, a surprisingly large number of tasks can be accomplished just by looking at word counts, and for those bag of words is sufficient. Some advantages of bag of words are that it is light on memory, and fast to train if you have your dictionary already. 

<br/>

**What are N-grams? How can we use them? ‍⭐️**

N-grams are phrases consisting of N-many words. Creating a bag of N-grams instead of a bag of words is a good way to overcome some of the weaknesses of bag of words. N-grams are a way of incorporating more context. They are also a way of picking up single concepts that don't fit in one word narrowly defined, such as 'San Francisco' or 'witch hunt'. Those phrases mean something different when they occur unified, which a normal bag of words won't pick up on. 

<br/>


**How large should be N for our bag of words when using N-grams? ‍⭐️**

Probably no bigger than 3 or 4 at the absolute maximum. Very few quartets of words occur so commonly together that it makes sense to treat them this way--namely, more or less as a single word. N is called the range of an N-gram. 

<br/>

**What is TF-IDF? How is it useful for text classification? ‍⭐️**

TF-IDF is a more sophisticated strategy (than Bag of Words) for representing a document as a numeric vector. As with Bag of Words, TF-IDF begins with a pre-determined, ordered vocabulary. Then, for each document, each word in that document is assigned a TF-IDF score (relative to that document) in the manner described below; then the document is represented as the vector of TF-IDF scores of each word in the vocabulary for that document. 

If w is a word and d is a document, the TF-IDF score TF-IDF(w,d) of w relative to d is: 
TF-IDF(w,d) = TF(w,d) * IDF(w),

where: 

TF(w,d) = is the count of occurrences of w in d divided by the number of words in d, and 
IDF(w) is the *inverse* of (a measure of) the *frequency* of w among the documents in the corpus. 

More specifically (defining IDF(w)): 

Let DF(w) = the number of times w occurs in the documents in the corpus (all taken together), and N be the number of documents in the corpus. Then 

IDF(w) = log(N/(DF(w) + 1))

We divide N by DF(w) to normalize (scaling our measure of inverse frequency by the number of documents). We take the log because as N increases, N/DF(w) explodes. And we add 1 so that we never take the log of 0, even when DF(w) = 0. 

<br/>



**Which model would you use for text classification with bag of words features? ‍⭐️**

You could use any of several classifiers that take numeric vectors as inputs: logistic regression, tree-based classifiers, support vector machines, etc. 

<br/>

**Would you prefer gradient boosting trees model or logistic regression when doing text classification with bag of words? ‍⭐️**

One should generally start with linear models unless domain knowledge gives one reasons to believe it is an inappropriate tool. Indeed, it is common for people to use logistic regression for text classification. 

However, here's an argument that tree-based methods can be appropriate also: 
Logistic regression makes the most sense when the predictors are numeric variables, specifically real numbers. Gradient boosting is preferable to logistic regression when the predictors are categorical. Now, there is a subtlety here, because  but that is not the case with bag of words; each predictor is numerical. On the other hand, since the predictor values are nonnegative integers of limited size (word counts in a document), perhaps they can reasonably be treated as high-cardinality categorical variables. 

<br/>

**What are word embeddings? Why are they useful? Do you know Word2Vec? ‍⭐️**

A word embedding is a mapping of words into a vector space of real numbers. Word embeddings are useful because they allow textual data to be processed and analyzed using algorithms that only accept numerical inputs. 

Word2vec is a group of related models that are used to produce word embeddings. These models are two-layer neural networks that are trained to reconstruct linguistic contexts of words. Words are then represented as vectors in a vector space typically consisting of several hundred dimensions. There are several ways that similarity of words can be defined in word2vec, but cosine similarity is a very common one. 

Word2vec learns the position of a word in the vector space by considering how it relates to other words that neighbor it in the corpus. That can be done in one of two ways, either using context to predict a target word (a method known as continuous bag of words, or CBOW), or using a word to predict a target context, which is called skip-gram. Using skip-gram, for example, when the feature vector assigned to a word cannot be used to accurately predict that word’s context, the components of the vector are adjusted. Each word’s context in the corpus is the "teacher" sending error signals back to adjust the feature vector. The vectors of words that are judged similar by their context are nudged closer together by adjusting the numerical entries in the vector.

<br/>

**Do you know any other ways to get word embeddings? 🚀**

Answer here

<br/>

**If you have a sentence with multiple words, you may need to combine multiple word embeddings into one. How would you do it? ‍⭐️**

Answer here

<br/>

**Would you prefer gradient boosting trees model or logistic regression when doing text classification with embeddings? ‍⭐️**

Answer here

<br/>

**What is the difference between a sequence model and an N-gram model?  ‍⭐️**

NLP models can be broadly classified into two categories: those that use word ordering information (sequence models), and ones that just see text as “bags” (sets) of words (n-gram models). Types of sequence models include convolutional neural networks (CNNs), recurrent neural networks (RNNs), and their variations. Types of n-gram models include logistic regression, simple multi- layer perceptrons (MLPs, or fully-connected neural networks), gradient boosted trees and support vector machines.

<br/>

**How can you use neural nets for text classification? 🚀**

Answer here

<br/>

**How can we use CNN for text classification? 🚀**

Answer here

<br/>


## Clustering

**What is unsupervised learning? 👶**

Unsupervised learning is when there are no labels attached to the data. You do not know *a priori* which data fall into which groups.

Some examples of unsupervised learning are autoencoders, clustering, topic modeling (LDA), and dimensionality reduction. 

Why are those unsupervised learning? It's because you are trying to find patterns in your data without having an input target. 

In an autoencoder, you convert your data to a vector. You don't need to label your data. 

Topic modeling is also unsupervised, since you don't know the topic for each document. The documents don't have labels. 

<br/>

**What is clustering? When do we need it? 👶**

Clustering is a tool to group like observations together. We need it because grouping like objects together can give us hints as to the label we should assign to that data.

Here's an example. Imagine that you work at Honda. You have craigslist data of car listings from all over the country. There's a strong demand for Honda in Des Moines. Your goal is to find similar markets. So you can use clustering to identify other listings similar to the Des Moines listing. Then you can target the other markets in the same cluster as the Des Moines listing. 

<br/>

**Do you know how K-means works? ‍⭐️**

You start by randomly assigning k "mean values" in your (potentially high dimensional) data space. You then associate every single data point with the nearest "mean" point. Then you compute the centroid of all the data associated with the individual mean. This becomes you new "mean" and you rinse and repeat until convergence or the specified number of iterations is completed.

<br/>

**How to select K for K-means? ‍⭐️**

You should run k-means several times with different numbers of clusters. Then you look at something called the *explained variance*, which measures how well the model accounts for the variation in the data. Use something like the "elbow method" to try to maximize this.

See all [this wikipedia page](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set).
<br/>

**What are the other clustering algorithms do you know? ‍⭐️**

KNN, Optics, Hierarchical clustering, DBScan, Hierarchical DBScan (HDBScan), Ward clustering, GMM

In GMM, each point has a probability of belonging in each cluster, rather than flatly assigned to a cluster. 

https://dev.tube/video/dGsxd67IFiU 

https://towardsdatascience.com/gaussian-mixture-modelling-gmm-833c88587c7f 
<br/>

**Do you know how DBScan works? ‍⭐️**

In DBScan, you define two points: the distance between points, and then number of neighborhood points. Each point has its neighborhood that you base on this distance. Core points are the ones that have a lot of points in their neighborhoods. Each point has its neighborhood-- all pts reachable from it-- within that distance. If your neighborhood is bigger than the threshold then you're a core point. 

Core points that are connected with each other count as in the same cluster. Other points that are reachable from a core point's cluster count as being in that core point's cluster. If your core point is A, and you want to determine whether B is in A's cluster, ask whether A and B are density connected. If there's a point C that's a core point and both A and B are within the threshold distance of C, then assign B to A's cluster. 


<br/>

**When would you choose K-means and when DBScan? ‍⭐️**

You can choose K-means when you know how many clusters you want. It is also pretty fast. But Kmeans only works for data clusters that are circle-ish shaped, since you're choosing clusters based on the distance to the centroid. 

DBScan doesn't require the number of clusters, but does have other hyperparameters which would need to be tuned. DBScan can be used to can find outliers, because it doesn't necessarily assign every point to a cluster. Also, it can deal with weirdly-shaped clusters. E.g., if you have a ball and a ring around it, Kmeans doesn't know what to do with that, but DBScan can tell you. 

<br/>

**How do you make this decision when you can't visualize your data because it's too high dimensional?  ‍⭐️**

If you have an intuition about what your data is like, go to the right algorithm. If not, it's ok to start with a simple algorithm and build up from there. Also: density-based algorithms can get tricked into placing things all into the same cluster, if the data points are connected by dense corridors. (Imagine a barbell with a dense bar.) 

<br/>

## tSNE
**What is tSNE and how does it work? 
⭐️**

tSNE is an algorithm that projects higher-dimensional data onto two dimensions in a way that preserves clusters (though not necessarily the orientation of clusters to one another). It is a good way to produce visualizations of clusters in higher-dimensional data.

<br/>


## Dimensionality reduction
**What is the curse of dimensionality? Why do we care about it? ‍⭐️**

Answer here

<br/>

**Do you know any dimensionality reduction techniques? ‍⭐️**

PCA
<br/>

**What’s singular value decomposition? How is it typically used for machine learning? ‍⭐️**

Suppose our data consists of an n x p matrix $$\ X $$. Singular value decomposition rewrites $$\ X $$ as:

$$\ X = UDV^T $$,

where U is an n x p orthogonal matrix ($$\ U^{T}U = I_p), V is a p x p orthogonal matrix, and D is a p x p diagonal matrix whose diagonal entries are all >= 0. The columns of U are known as the *left singular vectors*, the columns of V are known as the *right singular vectors*, and the diagonal entries of D are known as the *singular values*. (Note: the columns of UD are called the *principal components* of X.)

Singular value decomposition is used for both PCA and collaborative filtering. 

https://www.youtube.com/watch?v=P5mlg91as1c&feature=youtu.be

<br/>


**Do different matrix factorization techniques yield the same results? If so, why, and if not, why not?  ‍⭐️**

Generous Hint: Compare non-negative matrix factorization, the technique used for collaborative filtering, vs something like PCA.

<br/>


**What is Latent Dirichlet Allocation for, and how does it work? ‍⭐️**

Latent Dirichlet Allocation (LDA) is an NLP technique that is used for topic modeling. It was developed by David Blei, Andrew Ng, and Michael Jordan in [this](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) paper. Blei describes it in detail [here](http://videolectures.net/mlss09uk_blei_tm/). 

LDA assumes that each of the documents in the input corpus are generated by some small number of topics, out of a set of topics with a pre-set size. The goal is to identify which topics these are-- that is, for each document, to identify which topics were likeliest to have generated it. LDA does this by treating each document as a bag of words, and then calculating which topics are likeliest to have generated the words that each document contains, in the frequencies in which that document contains them. 

More precisely, the model assumes that for each document i there is a certain distribution theta_i of topics within that document, and within each topic k there is a certain distribution phi_k of words within that topic. Then we assume that each document i in the corpus is generated by repeatedly drawing topics k from distribution theta_i and then words from distribution phi_k until we've reached the number of words in that document. 

(The term 'Dirichlet' comes from the fact that the distribution of topics within documents, as well as the distribution of words within topics, are both assumed to be sparse Dirichlet distributions (symmetric parameter < 1). This corresponds to the assumption that any one document is likely to concern (be generated by) only a small number of topics; e.g., it is unlikely for a document to be about sports, politics, science, fashion, and literature all at once! Likewise, for each topic, only certain words are likely to be strongly associated with (generated by) that topic; the vast majority of words are fairly unrelated, though they might still have a very small likelihood of being used in connection with that topic, in certain contexts.)

Once we assume the corpus has been generated in the way just described, the challenge is then to learn the various distributions (the set of topics, their associated word probabilities, the topic of each word, and the particular topic mixture of each document). There have been several approaches to this; the one I am aware of uses Markov Chain Monte Carlo with Gibbs Sampling [(Griffiths and Steyvers 2004)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC387300/pdf/1015228.pdf). 

**Gibbs Sampling**:

Here is a quick explanation of how that works, inspired by [this video](https://www.youtube.com/watch?v=BaM1uiCpj_E).
Note first that the goal here is to assign each word a probability in each topic in a way that simultaneously satisfies the following two conditions: 
1. Each document is generated by as few topics as possible. 
2. Each word belongs to (has a high probability in) as few topics as possible. 

Given that goal, Gibbs sampling amounts to the following strategy:
i) Initialize by, for each word w and each topic i, randomly assigning w a probability in topic i. 
i) Now randomly choose a word w in a given document d. 
ii) Assume (probably falsely!) that all the other words have been correctly assigned their probabilities in each topic. 
iii) For each topic i, assign the instance of w in document d the following probability of having been generated by topic i: 
      (probability of topic i in document d) * (probability of w in topic i, based on other instances of w). 
      Here the probabilities are determined by the Dirichlet priors. 
iv) Repeat with randomly chosen other words until we've looked at all the words. 
v) Repeat some given number of times, looping through the entire corpus. 

**\*\*\***

LDA models are evaluated via two measures, *perpexity* and *coherence score*. Perplexity is a standard measure of performance for statistical models  of  natural  language. It measures the uncertainty in predicting a single word; lower values are better, and chance performance results in a perplexity equal to the size of the vocabulary. 

Coherence score measures the quality of the learned topics. The higher the score is, the better job has been done extracting the topics. Coherence is used (via cross validation) to determine the optimum number of topics to be extracted using LDA. You find the K that maximizes the sum, from 1 to K, of coherence scores for topics 1 to K. There is more than one way to define coherence, but the one I'm familiar with is [the UMass way](https://stats.stackexchange.com/questions/375062/how-does-topic-coherence-score-in-lda-intuitively-makes-sense). For two words w_i and w_j in the same topic, the score is higher if w_i and w_j appear together in documents a lot relative to how often wi alone appears in documents. This makes sense as a measure of topic coherence, since if two words in a "topic" that your model finds really belong together (that is, they really are about the same topic), you would expect them to show up together a lot. The denominator adjusts for the document frequency of the words you are considering, so that words like "the" don't get an artificially high score.

<br/>





## Ranking and search

**What is the ranking problem? Which models can you use to solve them? ‍⭐️**

Answer here

<br/>

**What are good unsupervised baselines for text information retrieval? ‍⭐️**

Answer here

<br/>

**How would you evaluate your ranking algorithms? Which offline metrics would you use? ‍⭐️**

Answer here

<br/>

**What is precision and recall at k? ‍⭐️**

Answer here

<br/>

**What is mean average precision at k? ‍⭐️**

Answer here

<br/>

**How can we use machine learning for search? ‍⭐️**

Answer here

<br/>

**How can we get training data for our ranking algorithms? ‍⭐️**

Answer here

<br/>

**Can we formulate the search problem as a classification problem? How? ‍⭐️**

Answer here

<br/>

**How can we use clicks data as the training data for ranking algorithms? 🚀**

Answer here

<br/>

**Do you know how to use gradient boosting trees for ranking? 🚀**

Answer here

<br/>

**How do you do an online evaluation of a new ranking algorithm? ‍⭐️**

Answer here

<br/>


## Recommender systems

**What is a recommender system? 👶**

Answer here

<br/>

**What are good baselines when building a recommender system? ‍⭐️**

Answer here

<br/>

**What is collaborative filtering? ‍⭐️**

You have a matrix where users are rows and items that you’re recommending are columns. For every user you have a record of the items they’ve shown interest in. You have this for every user. Collaborative filtering is where for each user, you look at other users who’ve watched the same shows as them, and then look at what they’ve watched. Different users are part of the matrix, but you have different holes. Predict what rating I would give to this movie, based on your similarity to other users who did watch the movie. 

The algorithm is matrix factorization. Singular value decomposition? 


https://en.wikipedia.org/wiki/Collaborative_filtering   

Could instead have an item-feature matrix (Pandora example): rows are items, columns are properties of items. 

Look up Netflix recommender system. 

<br/>

**How we can incorporate implicit feedback (clicks, etc) into our recommender systems? ‍⭐️**

Imagine we have a user-item table, where the items are movies showing each user's rating of that movie. 

Combine all these clicks somehow, and normalize it between 0 and 1, as a measure of the user's engagement with that movie. Then use that as a confidence score-- multiply the user rating by the engagement data-- elementwise multiplication of matrices. 

Could instead scale the user's *predicted* rating for that movie. 

http://activisiongamescience.github.io/2016/01/11/Implicit-Recommender-Systems-Biased-Matrix-Factorization/ 


<br/>

**What is the cold start problem? ‍⭐️**

How do you recommend a first thing to someone that you have no data on? Even if the website has existed already. 

<br/>

**Possible approaches to solving the cold start problem? ‍⭐️🚀**

Squeeze out as much information as you can: 
    • Use a subset of items and users that represent the population
    • Mine their social media presence
    • consider exploit vs explore-- how long do they click until you start recommending something
    • https://www.kdnuggets.com/2019/01/data-scientist-dilemma-cold-start-machine-learning.html
    • https://kojinoshiba.com/recsys-cold-start/

<br/>


## Time series

**What is a time series? 👶**

Answer here

<br/>

**How is time series different from the usual regression problem? 👶**

Answer here

<br/>

**Which models do you know for solving time series problems? ‍⭐️**

Answer here

<br/>

**If there’s a trend in our series, how we can remove it? And why would we want to do it? ‍⭐️**

Answer here

<br/>

**You have a series with only one variable “y” measured at time t. How do predict “y” at time t+1? Which approaches would you use? ‍⭐️**

Answer here

<br/>

**You have a series with a variable “y” and a set of features. How do you predict “y” at t+1? Which approaches would you use? ‍⭐️**

Answer here

<br/>

**What are the problems with using trees for solving time series problems? ‍⭐️**

Answer here

<br/>

**What are Monte Carlo simulations? ‍⭐️**
Monte Carlo simulations are a way of estimating a fixed parameter by repeatedly generating random numbers. By taking the random numbers generated and doing some computation on them, Monte Carlo simulations provide an approximation of a parameter where calculating it directly is impossible or prohibitively expensive. In practice, they’re used to forecast the weather, or estimate the probability of winning an election.

To begin, MCMC methods pick a random parameter value to consider. The simulation will continue to generate random values (this is the Monte Carlo part), but subject to some rule for determining what makes a good parameter value. The trick is that, for a pair of parameter values, it is possible to compute which is a better parameter value, by computing how likely each value is to explain the data, given our prior beliefs. If a randomly generated parameter value is better than the last one, it is added to the chain of parameter values with a certain probability determined by how much better it is (this is the Markov chain part).

A common use of Markov Chain Monte Carlo simulations is to estimate the posterior probability of a Bayesian inference problem. We know that the posterior distribution is somewhere in the range of our prior distribution and our likelihood distribution, but for whatever reason, we can’t compute it directly. Using MCMC methods, we’ll effectively draw samples from the posterior distribution, and then compute statistics like the average on the samples drawn.

https://towardsdatascience.com/understanding-monte-carlo-simulation-eceb4c9cad4
https://towardsdatascience.com/a-zero-math-introduction-to-markov-chain-monte-carlo-methods-dcba889e0c50
https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers
<br/>


**What are Markov Chains? ‍⭐️**

Andrey Markov sought to prove that non-independent events may also conform to patterns. One of his best known examples required counting thousands of two-character pairs from a work of Russian poetry. Using those pairs, he computed the conditional probability of each character. That is, given a certain preceding letter or white space, there was a certain chance that the next letter would be an A, or a T, or a whitespace. Using those probabilities, Markov was able to simulate an arbitrarily long over a few periods, can be used to compute the long-run tendency of that variable if we understand the probabilities that govern its behavior.

For generating markov chains: gibbs sampling or Metropolis-hastings algorithms
https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/

https://www.datasciencecentral.com/profiles/blogs/marketing-analytics-through-markov-chain

<br/>

**What is MCMC? ‍⭐️**

MCMC  == "the roomba problem"
MCMC methods are used to approximate the posterior distribution of a parameter of interest by random sampling in a probabilistic space.

https://en.wikipedia.org/wiki/Gibbs_sampling
http://www2.stat.duke.edu/~rcs46/modern_bayes17/lecturesModernBayes17/lecture-7/07-gibbs.pdf

http://nitro.biosci.arizona.edu/courses/EEB596/handouts/Gibbs.pdf

https://www.youtube.com/watch?v=h1NOS_wxgGg

Say you have probabilty of raining, sunny, and cloudy. Task is to predict the next state using probability of each. But you don't have the probability of each available. So you keep the probability of all variables except one constant and then estimate the probability of the remaining one. Then repeat for all the variables. 

<br/>

This project is a comprehensive and end-to-end classification algorithm on the customer churn data. At high level, this project is suitable to perform the data cleaning, data pre-processing, data visualization and finally build the classification model.

Machine learning is a branch of artificial intelligence (AI) and computer science that focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.

At its most basic, machine learning uses programmed algorithms that receive and analyse input data to predict output values within an acceptable range. As new data is fed to these algorithms, they learn and optimise their operations to improve performance, developing ‘intelligence’ over time.

Classical, or "non-deep," machine learning is more dependent on human intervention to learn. Human experts determine the set of features to understand the differences between data inputs, usually requiring more structured data to learn.

There are basically four types of machine learning algorithms: supervised, semi-supervised, unsupervised and reinforcement.


# 1) Supervised learning: Classification, Regression

In supervised learning, the machine is taught by example. The operator provides the machine learning algorithm with a known dataset that includes desired inputs and outputs, and the algorithm must find a method to determine how to arrive at those inputs and outputs. While the operator knows the correct answers to the problem, the algorithm identifies patterns in data, learns from observations and makes predictions. The algorithm makes predictions and is corrected by the operator – and this process continues until the algorithm achieves a high level of accuracy/performance.

Supervised learning, also known as supervised machine learning, is defined by its use of labeled datasets to train algorithms to classify data or predict outcomes accurately. As input data is fed into the model, the model adjusts its weights until it has been fitted appropriately. This occurs as part of the cross validation process to ensure that the model avoids overfitting or underfitting. Supervised learning helps organizations solve a variety of real-world problems at scale, such as classifying spam in a separate folder from your inbox. Some methods used in supervised learning include neural networks, naïve bayes, linear regression, logistic regression, random forest, and support vector machine (SVM).

In nutshell, the supervised learning can be furthered categorized into classification and regression algorithms. Classification model identifies which category an object belongs to whereas regression model predicts a continuous output.



# 2) Unsupervised machine learning: 

clustering algorithms (e.g., K Means Clustering, Hierarchical Clustering--dendrogram), association rule algorithms (e.g. Apriori), Dimensionality reduction

Unsupervised learning, also known as unsupervised machine learning, uses machine learning algorithms to analyze and cluster unlabeled datasets (subsets called clusters). These algorithms discover hidden patterns or data groupings without the need for human intervention. This method’s ability to discover similarities and differences in information make it ideal for exploratory data analysis, cross-selling strategies, customer segmentation, and image and pattern recognition. It’s also used to reduce the number of features in a model through the process of dimensionality reduction. Principal component analysis (PCA) and singular value decomposition (SVD) are two common approaches for this. Other algorithms used in unsupervised learning include neural networks, k-means clustering, and probabilistic clustering methods.

Thus, unsupervised learning reveals the underlying pattern in the dataset that are not explicitly presented, which can discover the similarity of data points (clustering algorithms) or uncover hidden relationships of variables (association rule algorithms).

# Examples of unsupervised learning algorithms include:

K-means clustering

Hierarchical clustering

Principal Component Analysis (PCA)

Association rule mining

Gaussian Mixture Models (GMM)

These algorithms are used to find patterns and structures within data without the need for labeled output.

# Difference between Supervised and Unsupervised learning:
![image](https://github.com/Tiwari666/Classification_Model/assets/153152895/5a130886-d0d5-4eeb-9ada-95f23f2b8ab3)



# 3) Semi-supervised learning: 

Semi-supervised learning offers a good medium between supervised and unsupervised learning. During training, it uses a smaller labeled data set to guide classification and feature extraction from a larger, unlabeled data set. Semi-supervised learning can solve the problem of not having enough labeled data for a supervised learning algorithm. It also helps if it’s too costly to label enough data. 

Examples of Semi-Supervised Learning:

a) Image classification: In image classification, the goal is to classify a given image into one or more predefined categories. Semi-supervised learning can be used to train an image classification model using a small amount of labeled data and a large amount of unlabeled image data.

b) Anomaly detection: In anomaly detection, the goal is to detect patterns or observations that are unusual or different from the normal condition.

c) Text classification: In text classification, the goal is to classify a given text into one or more predefined categories. Semi-supervised learning can be used to train a text classification model using a small amount of labeled data and a large amount of unlabeled text data.

d) Speech Analysis: Since labeling audio files is a very intensive task, Semi-Supervised learning is a very natural approach to solve this problem.

e) Internet Content Classification: Labeling each webpage is an impractical and unfeasible process and thus uses Semi-Supervised learning algorithms. Even the Google search algorithm uses a variant of Semi-Supervised learning to rank the relevance of a webpage for a given query.

f) Protein Sequence Classification: Since DNA strands are typically very large in size, the rise of Semi-Supervised learning has been imminent in this field.

Please see the following atached diagram to know how the semi-supervised learning algorithms work:

![image](https://github.com/Tiwari666/Classification_Model/assets/153152895/ad9e3a5c-9a03-45bf-85ce-d8c5671d76d3)


# 4) Reinforcement machine learning: 

e.g., Q-learning (model-free algorithm), SARSA (model-free algorithm), TD learning (model-based algorithm), Monte Carlo method (model-free algorithm)

Reinforcement Learning (RL) is the science of decision making. It is about learning the optimal behavior in an environment to obtain maximum reward.

Reinforcement machine learning is a machine learning model that is similar to supervised learning, but the algorithm isn’t trained using sample data. This model learns as it goes by using trial and error. A sequence of successful outcomes will be reinforced to develop the best recommendation or policy for a given problem.

Example:

Predictive text, text summarization, question answering, and machine translation are all examples of natural language processing (NLP) that uses reinforcement learning. By studying typical language patterns, RL agents can mimic and predict how people speak to each other every day.


# What machine-learning algorithms to use?

Choosing the right machine learning algorithm depends on several factors, including, but not limited to: data size, quality and diversity, as well as what answers businesses want to derive from that data. Additional considerations include accuracy, training time, parameters, data points and much more. Therefore, choosing the right algorithm is both a combination of business need, specification, experimentation and time available. Even the most experienced data scientists cannot tell you which algorithm will perform the best before experimenting with others.

# List of Regressions:

A) Linear Regression : Linear regression is the most common method for supervised learning. We fit a regression line with the data points available. It is easy to interpret, cost-efficient, and is used as a baseline in any business case.

B) Polynomial Regression: 

C) Ridge Regression: Ridge regression is an improved version of linear regression. It removes some issues of the OLS (ordinary least squares) methodology. It also imposes a penalty for ranging coefficient values with the alpha parameter. This coefficient plays a vital role in the calculation of the residual sum of squares for ridge regression, making the model robust.

D) LASSO Regression: Modern data is often complex with non-linear patterns that cannot be modeled by simple linear models. Polynomial regressions are models where we fit a higher degree curve to the data. It makes the model more flexible and scalable. To implement this in scikit-learn, you have to use the pipeline component. You can define the polynomial degree required in the pipeline.

E) Decision Tree Based Regression: Decision tree regression is a tree-based model where the data is split into subgroups based on homogeneity. we can import this model from the tree module of sklearn.

In order to avoid overfitting, make use of the “max_depth” parameter. It decides the maximum depth of the decision tree. If the value is set too high, the model might fit on noises and perform poorly upon a test dataset.

F) Random forest regression: Decision tree models are usually upscaled a level higher by combining multiple models. These are ensemble learning methods. They can be broadly classified into boosting and bagging algorithms.

The base models are weak learners, and by combining multiple weak learners, we get the final, strong learner model. The ‘ensemble’ module has all these functions in sklearn. “N_estimators” is an important parameter that decides the number of decision trees that require training.

G) Support Vector Regression (SVR): SVMs (support vector machines) were initially developed to classify problems, but they have been extended to apply to regression too. These models can be used when we have a higher dimension of features. They also provide different kernel options as per requirements.

# Key Metrics for Regression:

We need metrics to evaluate the outcome of a regression model. Therefore, the choice of an algorithm is that it will define which metric will be used to measure the performance. scikit-learn does not implement all performance metrics*

a) Mean Squared Error (MSE) — Average Square Error

b) Root Mean Squared Error (RMSE) — Square Root MSE

c) Mean Absolute Error (MAE) — Average Absolute Error

d) R Squared (R²) — Coefficient of Determination

e) Adjusted R Squared (R²) — R Adjusted

f) Mean Square Percentage Error (MSPE)

g) Mean Absolute Percentage Error (MAPE)

h) Root Mean Squared Logarithmic Error (RMSLE)



# Now, let's talk about the classification algorithm:

Classification is a natural language processing task that depends on machine learning algorithms.

Classification is a supervised machine learning method where the model tries to predict the correct label of a given input data. In classification, the model is fully trained using the training data, and then it is evaluated on test data before being used to perform prediction on new unseen data.

Classification is used when the target variable is categorical and discrete.

The goal of classification is to categorize input data into one of several predefined classes or labels.

In classification, the output is a label or category, representing a specific class or group that the input belongs to.

Common classification algorithms include logistic regression, decision trees, random forests, support vector machines, and neural networks.

Classification tasks include spam detection, sentiment analysis, image recognition, and medical diagnosis.

In summary, while regression predicts continuous numerical values, classification categorizes data into discrete classes or labels.

# Examples of classification algorithms: 

A) Binary classification: Logistic Regression , Decision Trees ,Naïve Bayes 
 Binary classification is when a model can apply only two class labels. A popular use of a binary classification would be in detecting and filtering junk emails. A model can be trained to label incoming emails as either junk or safe, based on learned patterns of what constitutes a spam email. 
 
B) Multiple class classification: Random Forest ,k-Nearest Neighbors, Naive Bayes 

Multiple class classification is when models reference more than the two class labels found in binary classification. Instead, there could be a huge array of possible class labels that could be applied to the object or data. An example would be in facial recognition software, where a model may analyse an image against a huge range of possible class labels to identify the individual. 


C) Multiple label classification: Multiple label Gradient Boosting , Multiple label Random Forests 

Multiple label classification is when an object or data point may have more than one class label assigned to it by the machine learning model. In this case the model will usually have multiple outputs. An example could be in image classification which may contain multiple objects. A model can be trained to identify, classify and label a range of subjects in one image. 

i) Logistic regression: This is a linear model, developed from linear regression to address classification issues. It uses the default regularization technique in the algorithm. When we apply this to multiclass classification problems, it uses the One vs Rest strategy.

ii) Support vector classifiers: SVM classifiers are popularly used for classification problems with a high dimension of features. They can transform the feature space into a higher dimension using the kernel function. Multiple kernel options are available including linear, RBF (radial base function), polynomial, and so on. We can also finetune the ‘gamma’ parameter, which is the kernel coefficient.

iii) Naive Bayes classifier
The gaussian Naive Bayes is a popular classification algorithm. It applies Bayes’ theorem of conditional probability to the case. It assumes that the features are independent of each other, while the targets are dependent on them. 

iv) Decision tree classifier
This is a tree-based structure, where a dataset is split based on values of various attributes. Finally, the data points with features of similar values are grouped together. Make sure to finetune the maximum depth and minimum leaf split parameters for better results. It also helps to avoid overfitting.

v) Gradient boosting classifier
Boosting is a method of ensemble learning where multiple decision trees are combined to enhance performance. It is a parallel learning method where multiple trees are trained parallelly and then combined to vote for the final result. We can finetune the hyperparameters like learning rate and number of estimators to achieve optimal training results.

vi) KNN classification
KNN (K nearest neighbor) is a classification algorithm that groups data points into clusters. The value of K can be chosen as a parameter “n_neighbors”. The algorithms form K clusters and assign each data point to the nearest cluster.

KNN performs multiple iterations where the distance of the points are the centers of the clusters, which are calculated and reassigned optimally.





# Metrics used for a classification algorithms: 

 For a classification problem, several evaluation metrics can be utilized to assess the performance of a machine learning model. Some commonly used metrics include accuracy, precision, recall, F1-score, and area under the ROC curve (AUC-ROC).

Accuracy: It measures the proportion of correctly classified instances out of the total instances. However, it might not be suitable for imbalanced datasets.

Precision: It indicates the proportion of true positive predictions out of all positive predictions made by the model. It's useful when the cost of false positives is high.

Recall: It measures the proportion of true positive predictions out of all actual positive instances in the dataset. It's important when the cost of false negatives is high.

F1-score: It is the harmonic mean of precision and recall, providing a balance between the two metrics. It's useful when there is an uneven class distribution.

Area under the ROC curve (AUC-ROC): It evaluates the model's ability to discriminate between positive and negative classes across various threshold values. A higher AUC-ROC score indicates better performance.

The choice of evaluation metric depends on the specific characteristics of the dataset and the problem at hand. It's essential to consider the goals and requirements of the classification task to select the most appropriate metric for evaluation.

# ROC curve

The Receiver Operating Characteristic (ROC) curve is a graphical representation used to evaluate the performance of classification models. It plots the true positive rate (sensitivity) against the false positive rate (1 - specificity) at various threshold settings. ROC curves are useful because they provide a comprehensive understanding of a model's performance across different discrimination thresholds, allowing us to assess its trade-offs between sensitivity and specificity. A model with a higher area under the ROC curve (AUC) indicates better overall performance in distinguishing between the positive and negative classes.ROC curves are particularly valuable for comparing and selecting the best-performing model among multiple alternatives and for determining the optimal threshold for a given classification task.

# AUC-ROC

AUC-ROC, or Area Under the Receiver Operating Characteristic Curve, is a performance metric commonly used to evaluate the quality of a binary classification model. It measures the area under the curve plotted by the true positive rate (sensitivity) against the false positive rate (1-specificity) across different threshold values for classification decisions. AUC-ROC provides a single scalar value that represents the model's ability to discriminate between the positive and negative classes, with a higher value indicating better discrimination (a perfect classifier has an AUC-ROC score of 1). It is particularly useful for imbalanced datasets and provides a comprehensive assessment of the model's performance across various decision thresholds.

# confusion matrix

The confusion matrix is a performance evaluation tool used in classification tasks to visualize the performance of a machine learning model. It is a square matrix where rows represent the actual classes and columns represent the predicted classes. Each cell in the matrix represents the count of instances where the actual class (row) matches the predicted class (column). The confusion matrix provides valuable insights into the model's performance by breaking down predictions into four categories:

True Positive (TP): Instances where the model correctly predicts positive classes.

True Negative (TN): Instances where the model correctly predicts negative classes.

False Positive (FP): Instances where the model incorrectly predicts positive classes (Type I error).

False Negative (FN): Instances where the model incorrectly predicts negative classes (Type II error).

With this breakdown, various performance metrics such as accuracy, precision, recall (sensitivity), specificity, and F1-score can be calculated, aiding in assessing the model's effectiveness in classification tasks.













Sources:

Link 1: https://www.ibm.com/topics/machine-learning

Link2: https://www.sas.com/en_gb/insights/articles/analytics/machine-learning-algorithms.html

Link 3: https://towardsdatascience.com/top-machine-learning-algorithms-for-classification

Link: https://www.quora.com/What-are-some-of-the-well-known-reinforcement-learning-algorithms

Link 5: https://monkeylearn.com/blog/classification-algorithms/

Link 6: https://www.v7labs.com/blog/supervised-vs-unsupervised-learning#what-is-semi-supervised-learning

Link 7: https://www.seldon.io/supervised-vs-unsupervised-learning-explained

Link 8: https://www.turing.com/kb/scikit-learn-cheatsheet-methods-for-classification-and-regression




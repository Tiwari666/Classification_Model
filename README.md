This project is a comprehensive and end-to-end classification algorithm on the customer churn data. At high level, this project is suitable to perform the data cleaning, data pre-processing, data visualization and finally build the classification model.

Machine learning is a branch of artificial intelligence (AI) and computer science that focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.

At its most basic, machine learning uses programmed algorithms that receive and analyse input data to predict output values within an acceptable range. As new data is fed to these algorithms, they learn and optimise their operations to improve performance, developing ‘intelligence’ over time.

Classical, or "non-deep," machine learning is more dependent on human intervention to learn. Human experts determine the set of features to understand the differences between data inputs, usually requiring more structured data to learn.

There are basically four types of machine learning algorithms: supervised, semi-supervised, unsupervised and reinforcement.


# 1) Supervised learning: Classification, Regression

In supervised learning, the machine is taught by example. The operator provides the machine learning algorithm with a known dataset that includes desired inputs and outputs, and the algorithm must find a method to determine how to arrive at those inputs and outputs. While the operator knows the correct answers to the problem, the algorithm identifies patterns in data, learns from observations and makes predictions. The algorithm makes predictions and is corrected by the operator – and this process continues until the algorithm achieves a high level of accuracy/performance.

Supervised learning, also known as supervised machine learning, is defined by its use of labeled datasets to train algorithms to classify data or predict outcomes accurately. As input data is fed into the model, the model adjusts its weights until it has been fitted appropriately. This occurs as part of the cross validation process to ensure that the model avoids overfitting or underfitting. Supervised learning helps organizations solve a variety of real-world problems at scale, such as classifying spam in a separate folder from your inbox. Some methods used in supervised learning include neural networks, naïve bayes, linear regression, logistic regression, random forest, and support vector machine (SVM).



# 2) Unsupervised machine learning: 

clustering algorithms (e.g., K Means Clustering, Hierarchical Clustering--dendrogram), association rule algorithms (e.g. Apriori), Dimensionality reduction

Unsupervised learning, also known as unsupervised machine learning, uses machine learning algorithms to analyze and cluster unlabeled datasets (subsets called clusters). These algorithms discover hidden patterns or data groupings without the need for human intervention. This method’s ability to discover similarities and differences in information make it ideal for exploratory data analysis, cross-selling strategies, customer segmentation, and image and pattern recognition. It’s also used to reduce the number of features in a model through the process of dimensionality reduction. Principal component analysis (PCA) and singular value decomposition (SVD) are two common approaches for this. Other algorithms used in unsupervised learning include neural networks, k-means clustering, and probabilistic clustering methods.

Thus, unsupervised learning reveals the underlying pattern in the dataset that are not explicitly presented, which can discover the similarity of data points (clustering algorithms) or uncover hidden relationships of variables (association rule algorithms).

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

# Now, let's talk about the classification algorithm:



















Sources:

Link 1: https://www.ibm.com/topics/machine-learning

Link2: https://www.sas.com/en_gb/insights/articles/analytics/machine-learning-algorithms.html

Link 3: https://towardsdatascience.com/top-machine-learning-algorithms-for-classification

Link: https://www.quora.com/What-are-some-of-the-well-known-reinforcement-learning-algorithms

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# Generate Data with n=100 std deviation=2 and 2 clusters
X,y = make_blobs(n_samples=100, cluster_std=7, centers=2)

# splitting the test and train data using a 30% splitting criterion
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1)

# training a naive bayes classifier
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

# testing the accuracy
y_pred_train =naive_bayes.predict(X_train)
y_pred_test = naive_bayes.predict(X_test)
print('Accuracy_test:%2f'%metrics.accuracy_score(y_test,y_pred_test))
print('Accuracy_train:%2f'%metrics.accuracy_score(y_train,y_pred_train))
# Train Accuracy: 1.00 & Test Accuracy:1.00

# plotting on scatter plot
colors = ['r' if i==0 else 'b' for i in y]
X0,X1 = X.T
plt.scatter(X0, X1, c=colors)
plt.show()
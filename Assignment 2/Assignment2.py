from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import model_selection

wine = datasets.load_wine()
# Display Features
print(wine.feature_names)

# Display Target
print(wine.target_names)

# Obtaining Features & Target
Feature = wine.feature_names
Target = wine.target_names

X,y = datasets.load_wine(return_X_y = True)
print(type(X), type(y))
print(X.shape, y.shape)

X_train, X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.30, random_state=1)
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

y_pred_test = dtree.predict(X_test)
y_pred_train = dtree.predict(X_train)

print('Accuracy_test:%2f'%metrics.accuracy_score(y_test,y_pred_test))
print('Accuracy_train:%2f'%metrics.accuracy_score(y_train,y_pred_train))

# Train accuracy is better
# Support Vector Machines and Decision Trees
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)
print("Linear SVM accuracy:", linear_svm.score(X_test, y_test))

kernel_svm = SVC(kernel='rbf')
kernel_svm.fit(X_train, y_train)
print("Kernel SVM accuracy:", kernel_svm.score(X_test, y_test))

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
print("Decision Tree accuracy:", decision_tree.score(X_test, y_test))

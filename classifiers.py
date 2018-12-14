from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB

classifiers = [
    {
        'name': "Nearest Neighbors k=3",
        'model': KNeighborsClassifier(3)
    },
    {
        'name': "Nearest Neighbors k=30",
        'model': KNeighborsClassifier(30)
    },
    {
        'name': "Nearest Neighbors k=100",
        'model': KNeighborsClassifier(100)
    },
    {
        'name': "Linear SVM",
        'model': SVC(kernel="linear", C=0.025)
    },
    {
        'name': "Decision Tree",
        'model': DecisionTreeClassifier(max_depth=5)
    },
    {
        'name': "Random Forest",
        'model': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    },
    {
        'name': "Naive Bayes",
        'model': MultinomialNB()
    },
    {
        'name': "Logistic Regression",
        'model': LogisticRegression()
    }
]
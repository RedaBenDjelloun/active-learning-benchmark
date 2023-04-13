from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from generators.generator import GaussianGenerator, DataGenerators
from matplotlib import pyplot as plt
import numpy as np

dim = 200
classifier = LogisticRegression

std = 0.5
target = np.zeros(dim)
target[0] = 1

basic_train = 10
size_train = 1000
size_val = 1000
max_size = 400

std = np.ones(dim)
std[0] = 0.2

# Construct generator
lst_gen = []
lst_gen.append(GaussianGenerator(target,std))
lst_gen.append(GaussianGenerator(-target,std))
data_gen = DataGenerators(lst_gen)

# Construct active learner
learner = ActiveLearner(
    estimator=classifier(),
    query_strategy=uncertainty_sampling
)

# Generate dataset
X_train, y_train = data_gen.generate_data(size_train)
X_val, y_val = data_gen.generate_data(size_val)

def compute_accuracies(X_train, y_train, X_val, y_val, max_size):
    # Initial teaching
    learner.teach(X_train[:basic_train], y_train[:basic_train])
    X_left, y_left = X_train[basic_train:], y_train[basic_train:]
    # Initialization
    clf = classifier()
    clf.fit(X_train[:basic_train],y_train[:basic_train])
    accuracies_learner = [learner.score(X_val,y_val)]
    accuracies_basic = [clf.score(X_val,y_val)]
    # Loop over examples
    for i in  range(basic_train,max_size):
        # Obtain query
        query_idx, _ = learner.query(X_left)
        # Perform a step of active learning
        learner.teach(X_left[query_idx], y_left[query_idx])
        # Update remaining data
        X_left = np.delete(X_left, query_idx,0)
        y_left = np.delete(y_left,query_idx,0)
        # Recreate classifier
        clf = classifier()
        clf.fit(X_train[:i],y_train[:i])
        # Add accuracies to array
        accuracies_learner.append(learner.score(X_val,y_val))
        accuracies_basic.append(clf.score(X_val,y_val))
    return accuracies_basic, accuracies_learner

x = [i for i in range(basic_train,max_size+1)]

accuracies_basic, accuracies_learner = compute_accuracies(X_train, y_train, X_val, y_val, max_size)

plt.plot(x,accuracies_basic)
plt.plot(x,accuracies_learner)
plt.legend(["basic","learner"])
plt.show()
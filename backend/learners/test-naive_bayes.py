from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.naive_bayes import GaussianNB
from generators.generator import GaussianGenerator, DataGenerators
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import numpy as np

dim = 100
basic_train = 10
size_train = 100
size_val = 1000
classifier = GaussianNB
target = np.zeros(dim)
target[0] = 1
# Construct generator
lst_gen = []
lst_gen.append(GaussianGenerator(target,1))
lst_gen.append(GaussianGenerator(-target,1))
data_gen = DataGenerators(lst_gen)

# Construct active learner
learner = ActiveLearner(
    estimator=classifier(),
    query_strategy=uncertainty_sampling
)

X_train, y_train = data_gen.generate_data(size_train)
X_val, y_val = data_gen.generate_data(size_val)

learner.teach(X_train[:basic_train], y_train[:basic_train])
X_left, y_left = X_train[basic_train:], y_train[basic_train:]

accuracies_learner = []
accuracies_basic = []
x = [i for i in range(basic_train+1,size_train+1)]
for i in  range(basic_train,size_train):
    query_idx, _ = learner.query(X_left)
    learner.teach(X_left[query_idx], y_left[query_idx])
    X_left = np.delete(X_left, query_idx,0)
    y_left = np.delete(y_left,query_idx,0)
    clf = classifier()
    clf.fit(X_train[:i],y_train[:i])
    accuracies_learner.append(learner.score(X_val,y_val))
    accuracies_basic.append(clf.score(X_val,y_val))

#plt.plot(x,accuracies_basic)
#plt.plot(x,accuracies_learner)
#plt.legend(["basic","learner"])
#plt.show()

def plot(classifier):
    learner = ActiveLearner(
    estimator=classifier(),
    query_strategy=uncertainty_sampling
)
    learner.teach(X_train[:basic_train], y_train[:basic_train])
    X_left, y_left = X_train[basic_train:], y_train[basic_train:]

    accuracies_learner = []
    accuracies_basic = []
    x = [i for i in range(basic_train+1,size_train+1)]
    for i in  range(basic_train,size_train):
        query_idx, _ = learner.query(X_left)
        learner.teach(X_left[query_idx], y_left[query_idx])
        X_left = np.delete(X_left, query_idx,0)
        y_left = np.delete(y_left,query_idx,0)
        clf = classifier()
        clf.fit(X_train[:i],y_train[:i])
        accuracies_learner.append(learner.score(X_val,y_val))
        accuracies_basic.append(clf.score(X_val,y_val))

    plt.plot(x,accuracies_basic)
    plt.plot(x,accuracies_learner)
    #plt.legend(["basic","learner"])
    #plt.show()


if __name__ == "__main__":
    figure =plt.figure(figsize=(15,11))
    plot(GaussianNB)
    plot(LogisticRegression)
    plt.legend(["basic_NB","learner_NB","basic_Logit","learner_Logit"])
    plt.show()



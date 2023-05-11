from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from generators.generator import GaussianGenerator, DataGenerators, not_convex
from matplotlib import pyplot as plt
import numpy as np
from comparator import generate_dataset, plot_accuracies, construct_learner

def compute_time_accuracies(data_gen, size_train, size_val, basic_train, nb_queries_by_step, nb_steps, learner, classifier, dt):
    t=0
    X_train_basic, y_train_basic, X_val, y_val = generate_dataset(data_gen,basic_train,size_val,t)
    # Initial teaching
    learner.teach(X_train_basic,y_train_basic)
    # Initialization
    clf = classifier()
    clf.fit(X_train_basic,y_train_basic)
    accuracies_learner = [learner.score(X_val,y_val)]
    accuracies_basic = [clf.score(X_val,y_val)]

    for i in range(nb_steps):
        t+=dt
        X_train, y_train, X_val, y_val = generate_dataset(data_gen,size_train,size_val,t)
        X_train_basic = np.concatenate((X_train_basic, X_train[:nb_queries_by_step]))
        y_train_basic = np.concatenate((y_train_basic, y_train[:nb_queries_by_step]))
        # Obtain query
        for i in range(nb_queries_by_step):
            query_idx, _ = learner.query(X_train)   
            # Perform a step of active learning
            learner.teach(X_train[query_idx], y_train[query_idx])
            # Update remaining data
            X_train = np.delete(X_train, query_idx,0)
            y_train = np.delete(y_train,query_idx,0)
        # Recreate classifier
        clf = classifier()
        clf.fit(X_train_basic,y_train_basic)
        # Add accuracies to array
        accuracies_learner.append(learner.score(X_val,y_val))
        accuracies_basic.append(clf.score(X_val,y_val))
    return accuracies_basic, accuracies_learner

def plot_time_accuracies(ax,accuracies_basic, accuracies_learner, dt, nb_steps):
    x = [i*dt for i in range(nb_steps+1)]
    ax.plot(x,accuracies_basic)
    ax.plot(x,accuracies_learner)
    ax.legend(["basic","learner"])


if __name__ == "__main__":
    dim = 50
    classifier = LogisticRegression
    query_strategy = uncertainty_sampling
    learner = construct_learner(classifier, query_strategy)

    target = np.zeros(dim)
    target[0] = 1
    depl = np.zeros(dim)
    depl[1] = 1

    basic_train = 10
    size_train = 100
    size_val = 100
    dt = 0.1
    nb_queries_by_step = 1
    nb_steps = 100

    std = 0.6

    f1 = lambda t: (target,0.5*(1+np.sin(t)))
    f2 = lambda t: (-target,0.5*(1+np.sin(t)))
    # Construct generator
    lst_gen = []
    lst_gen.append(GaussianGenerator(target,std,updator=f1))
    lst_gen.append(GaussianGenerator(-target,std,updator=f2))
    data_gen = DataGenerators(lst_gen)


    accuracies_basic, accuracies_learner = compute_time_accuracies(data_gen, size_train, size_val, basic_train, nb_queries_by_step, nb_steps, learner, classifier, dt)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    plot_time_accuracies(ax, accuracies_basic, accuracies_learner, dt, nb_steps)
    plt.show()


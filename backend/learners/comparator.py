from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from generators.generator import GaussianGenerator, DataGenerators, not_convex

from matplotlib import pyplot as plt
import numpy as np

# Construct generator
def create_two_gaussians(dim = 2, first_dim_mean = 1, first_dim_std = 0.2, other_dim_stds=1):
    target = np.zeros(dim)
    target[0] = first_dim_mean
    std = np.ones(dim)*other_dim_stds
    std[0] = first_dim_std
    lst_gen = []
    lst_gen.append(GaussianGenerator(target,std))
    lst_gen.append(GaussianGenerator(-target,std))
    data_gen = DataGenerators(lst_gen)
    return data_gen

# Construct active learner
def construct_learner(classifier, query_strategy):
    return ActiveLearner(
        estimator=classifier(),
        query_strategy=query_strategy
    )

# Generate dataset
def generate_dataset(data_generator, size_train, size_val, t=None):
    X_train, y_train = data_generator.generate_data(size_train,t)
    X_val, y_val = data_generator.generate_data(size_val,t)
    return X_train, y_train, X_val, y_val

def compute_accuracies(X_train, y_train, X_val, y_val, basic_train, nb_queries, learner, classifier):
    # Initial teaching
    learner.teach(X_train[:basic_train], y_train[:basic_train])
    X_left, y_left = X_train[basic_train:], y_train[basic_train:]
    # Initialization
    clf = classifier()
    clf.fit(X_train[:basic_train],y_train[:basic_train])
    accuracies_learner = [learner.score(X_val,y_val)]
    accuracies_basic = [clf.score(X_val,y_val)]
    # Loop over examples
    for i in  range(basic_train,basic_train + nb_queries):
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

def plot_accuracies(ax, accuracies_basic,accuracies_learner,basic_train,nb_queries):
    x = [i for i in range(basic_train,basic_train+nb_queries+1)]
    ax.plot(x,accuracies_basic)
    ax.plot(x,accuracies_learner)
    ax.legend(["basic","learner"])

def do_benchmark(ax, data_generator, classifier, query_strategy, basic_train, nb_queries, size_train, size_val):
    data_generator = data_generator
    learner = construct_learner(classifier, query_strategy)
    X_train, y_train, X_val, y_val = generate_dataset(data_generator, size_train, size_val)
    accuracies_basic, accuracies_learner = compute_accuracies(X_train, y_train, X_val, y_val, basic_train, nb_queries, learner, classifier)
    plot_accuracies(ax, accuracies_basic,accuracies_learner,basic_train,nb_queries)

def create_SVC():
    return SVC(kernel="linear", C=0.025, probability=True)


if __name__ == "__main__":
    dim = 100
    data_generator = create_two_gaussians(dim=dim, first_dim_mean=1, first_dim_std=0.5, other_dim_stds=1)
    #data_generator = not_convex(dim)
    classifier = LogisticRegression
    query_strategy=uncertainty_sampling

    basic_train = 10
    nb_queries = 290
    size_train = 1000
    size_val = 1000

    #do_benchmark(data_generator, classifier, query_strategy, basic_train, nb_queries, size_train, size_val)

    dim_values = [20,50,100,200]
    classifier_values = [LogisticRegression, GaussianNB, create_SVC]
    classifier_names = ["LogisticRegression", "GaussianNB", "SVC"]

    fig = plt.figure()
    for (clf_idx, classifier) in enumerate(classifier_values):
        for (dim_idx, dim) in enumerate(dim_values):
            data_generator = create_two_gaussians(dim=dim, first_dim_mean=1, first_dim_std=0.5, other_dim_stds=1)
            ax = fig.add_subplot(3,4,1+dim_idx+clf_idx*4)
            ax.set_title("dim = " + str(dim) + ", clf = " + classifier_names[clf_idx])
            do_benchmark(ax, data_generator, classifier, uncertainty_sampling, basic_train, nb_queries, size_train, size_val)

    plt.tight_layout()
    plt.show()
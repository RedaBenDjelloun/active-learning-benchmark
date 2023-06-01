from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from generators.generator import GaussianGenerator, DataGenerators, DataGeneratorsWithHiddenFunction, UniformGenerator
from matplotlib import pyplot as plt
import numpy as np
from learners.comparator import generate_dataset, plot_accuracies, construct_learner, create_two_gaussians

def construct_water_level_data_generator(dim, amplitude = 0.3, frequency = 0.25, threshold = 0.5):
    def f(X,t): 
        return X[:,0]>threshold+amplitude*np.sin(2*np.pi*frequency*t)
    
    gen = UniformGenerator(np.zeros(dim),np.ones(dim))
    data_gen = DataGeneratorsWithHiddenFunction(gen,f,time_dependant_function=True)
    return data_gen

def compute_time_accuracies(data_gen, size_train, size_val, basic_train, nb_queries_by_step, nb_steps, learner, classifier, dt, gamma=1):
    t=0
    y_train_basic = np.array([])
    while len(np.unique(y_train_basic))<=1:
        X_train_basic, y_train_basic, X_val, y_val = generate_dataset(data_gen,basic_train,size_val,t)
    # Initial teaching
    learner.teach(X_train_basic,y_train_basic)
    # Initialization
    clf = classifier()
    clf.fit(X_train_basic,y_train_basic)
    X_train_learner = X_train_basic[:]
    y_train_learner = y_train_basic[:]
    accuracies_learner = [learner.score(X_val,y_val)]
    accuracies_basic = [clf.score(X_val,y_val)]
    weights = np.exp(-gamma*dt)*np.ones(basic_train)
    for i in range(nb_steps):
        t+=dt
        X_train, y_train, X_val, y_val = generate_dataset(data_gen,size_train,size_val,t)
        X_train_basic = np.concatenate((X_train_basic, X_train[:nb_queries_by_step]))
        y_train_basic = np.concatenate((y_train_basic, y_train[:nb_queries_by_step]))
        # Obtain query
        for i in range(nb_queries_by_step):
            weights = np.concatenate((weights,np.ones(1)))
            query_idx, _ = learner.query(X_train)
            X_train_learner =  np.concatenate((X_train_learner,X_train[query_idx]))
            y_train_learner =  np.concatenate((y_train_learner,y_train[query_idx]))
            # Perform a step of active learning
            #learner.fit(X_train_learner,y_train_learner,sample_weight=weights)
            # Update remaining data
            X_train = np.delete(X_train, query_idx, 0)
            y_train = np.delete(y_train, query_idx, 0)
        learner.fit(X_train_learner,y_train_learner,sample_weight=weights)
        # Recreate classifier
        clf = classifier()
        clf.fit(X_train_basic,y_train_basic,sample_weight=weights)
        # Add accuracies to array
        accuracies_learner.append(learner.score(X_val,y_val))
        accuracies_basic.append(clf.score(X_val,y_val))
        weights *= np.exp(-gamma*dt)
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
    size_train = 500
    size_val = 500
    dt = 0.1
    nb_queries_by_step = 5
    nb_steps = 200
    gamma = 0.8



    # Construct generator
    
    data_gen = construct_water_level_data_generator(dim)

    #data_gen = create_two_gaussians(dim=dim, first_dim_mean=1, first_dim_std=0.5, other_dim_stds=1)

    accuracies_basic, accuracies_learner = compute_time_accuracies(data_gen, size_train, size_val, basic_train, nb_queries_by_step, nb_steps, learner, classifier, dt, gamma)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    plot_time_accuracies(ax, accuracies_basic, accuracies_learner, dt, nb_steps)
    plt.show()


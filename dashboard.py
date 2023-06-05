import streamlit as st
import pandas as pd
import numpy as np

import numpy.random as rnd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px


from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from backend.generators.generator import GaussianGenerator, DataGenerators, not_convex, xor_gen
from backend.learners.comparator import create_two_gaussians, construct_learner, generate_dataset, compute_accuracies, plot_accuracies, do_benchmark, create_SVC
from backend.learners.time_dependancy import construct_water_level_data_generator, compute_time_accuracies, construct_shift_data_generator

st.sidebar.title('Active learning dashboard')

data_generator_names = ['Two Gaussians', 'Not convex', 'Water level', 'Shift', 'Xor']
classifier_names = ['Logistic Regression', 'Naive Bayes', 'SVM']
query_strategy_names = ['Uncertainty sampling', 'Margin sampling', 'Entropy sampling']

if "all_accuracies_learner" not in st.session_state:
    st.session_state.all_accuracies_learner = np.array([])
    st.session_state.all_accuracies_basic = np.array([])

def update_seed():
    rnd.seed(st.session_state.seed)

# Functions to update session state
def reinit_curve_data():
    st.session_state.all_accuracies_learner = np.array([])
    st.session_state.all_accuracies_basic = np.array([])

def update_data_generator():
    reinit_curve_data()
    if st.session_state.data_generator_name == 'Two Gaussians':
        st.session_state.data_generator = create_two_gaussians(dim=st.session_state.dimension, first_dim_mean=1, first_dim_std=0.5, other_dim_stds=1)
        st.session_state.data_generator2D = create_two_gaussians(dim=2, first_dim_mean=1, first_dim_std=0.5, other_dim_stds=1)
    elif st.session_state.data_generator_name == 'Not convex':
        st.session_state.data_generator = not_convex(dim=st.session_state.dimension)
        st.session_state.data_generator2D = not_convex(dim=2)
    elif st.session_state.data_generator_name == 'Water level':
        st.session_state.data_generator =  construct_water_level_data_generator(dim=st.session_state.dimension)
        st.session_state.data_generator2D = construct_water_level_data_generator(dim=2)
    elif st.session_state.data_generator_name == 'Xor':
        st.session_state.data_generator =  xor_gen(dim=st.session_state.dimension)
        st.session_state.data_generator2D = xor_gen(dim=2)
    elif st.session_state.data_generator_name == 'Shift':
        st.session_state.data_generator =  construct_shift_data_generator(dim=st.session_state.dimension)
        st.session_state.data_generator2D = construct_shift_data_generator(dim=2)

def update_classifier():
    reinit_curve_data()
    if st.session_state.classifier_name == 'Logistic Regression':
        st.session_state.classifier = LogisticRegression
    elif st.session_state.classifier_name == 'Naive Bayes':
        st.session_state.classifier = GaussianNB
    elif st.session_state.classifier_name == 'SVM':
        st.session_state.classifier = create_SVC

def update_query_strategy():
    reinit_curve_data()
    if st.session_state.query_strategy_name == 'Uncertainty sampling':
        st.session_state.query_strategy = uncertainty_sampling
    elif st.session_state.query_strategy_name == 'Margin sampling':
        st.session_state.query_strategy = margin_sampling
    elif st.session_state.query_strategy_name == 'Entropy sampling':
        st.session_state.query_strategy = entropy_sampling

# Add tabs in sidebar
tabs = ['Dataset', 'Classifier', 'Learner', 'Evolution']
sbtab1, sbtab2, sbtab3, sbtab4 = st.sidebar.tabs(tabs)


with sbtab1:
    # Slider for the seed
    if 'seed' not in st.session_state:
        st.session_state.seed = 42,
    
    seed = st.slider(
        'Seed',
        0, 100, step=1,
        key='seed',
        on_change=update_seed,
    )


    # Slider for the dimension
    if 'dimension' not in st.session_state:
        st.session_state.dimension = 100

    dimension = st.slider(
        'Dimension',
        10, 300, step=10, 
        key='dimension',
        on_change=update_data_generator
    )

    # Selectbox for the data generator
    if 'data_generator_name' not in st.session_state:
        st.session_state.data_generator_name = 'Two Gaussians'
    if 'data_generator' not in st.session_state:
        st.session_state.data_generator = create_two_gaussians(dim=st.session_state.dimension, first_dim_mean=1, first_dim_std=0.5, other_dim_stds=1)
    if 'data_generator2D' not in st.session_state:
        st.session_state.data_generator2D = create_two_gaussians(dim=2, first_dim_mean=1, first_dim_std=0.5, other_dim_stds=1)

    select_data_generator = st.selectbox(
        'Data generator',
        data_generator_names,
        key='data_generator_name',
        on_change=update_data_generator
    )

    # Slider for the size of the training set with initial value 1000
    if 'size_train' not in st.session_state:
        st.session_state.size_train = 1000

    size_train = st.slider(
        'Size of the training set',
        100, 3000, step=100, key='size_train',on_change=reinit_curve_data
    )

    # Slider for the size of the validation set with initial value 1000
    if 'size_val' not in st.session_state:
        st.session_state.size_val = 1000

    size_val = st.slider(
        'Size of the validation set',
        100, 3000, step=100, key='size_val',on_change=reinit_curve_data
    )

    if 'nb_replications_max' not in st.session_state:
        st.session_state.nb_replications_max = 5

    nb_replications_max = st.slider(
        'Number of replications',
        1, 20, step=1, 
        key='nb_replications_max',
    )
    


with sbtab2:
    # Selectbox for the classifier
    if 'classifier_name' not in st.session_state:
        st.session_state.classifier_name = 'Logistic Regression'
    if 'classifier' not in st.session_state:
        st.session_state.classifier = LogisticRegression

    select_classifier = st.selectbox(
        'Classifier',
        classifier_names,
        key='classifier_name',
        on_change=update_classifier
    )

with sbtab3:
    # Selectbox for the query strategy
    if 'query_strategy_name' not in st.session_state:
        st.session_state.query_strategy_name = 'Uncertainty sampling'
    if 'query_strategy' not in st.session_state:
        st.session_state.query_strategy = uncertainty_sampling

    select_query_strategy = st.selectbox(
        'Query strategy',
        query_strategy_names,
        key='query_strategy_name',
        on_change=update_query_strategy
    )

    # Slider for the number of basic train with initial value 10
    if 'basic_train' not in st.session_state:
        st.session_state.basic_train = 10

    basic_train = st.slider(
        'Number of basic train',
        2, 100, step=1, key='basic_train',on_change=reinit_curve_data
    )

    # Slider for the number of queries with initial value 290
    if 'nb_queries' not in st.session_state:
        st.session_state.nb_queries = 290

    nb_queries = st.slider(
        'Number of queries',
        10, 1000, step=10, key='nb_queries',on_change=reinit_curve_data
    )

with sbtab4:

    # Slider for the number of steps
    if 'nb_steps' not in st.session_state:
        st.session_state.nb_steps = 200
    
    nb_steps = st.slider(
        'Number of steps',
        10, 1000, step=10, key='nb_steps',on_change=reinit_curve_data
    )

    # Slider for the number of queries per step
    if 'nb_queries_per_step' not in st.session_state:
        st.session_state.nb_queries_per_step = 5
    
    nb_queries_per_step = st.slider(
        'Number of queries per step',
        1, 100, step=1, key='nb_queries_per_step',on_change=reinit_curve_data
    )

    # Slider for dt
    if 'dt' not in st.session_state:
        st.session_state.dt = 0.1
    
    dt = st.slider(
        'dt',
        0.0, 1.0, step=0.1, key='dt',on_change=reinit_curve_data
    )

    # Slider for number of points displayed
    if 'nb_points_displayed' not in st.session_state:
        st.session_state.nb_points_displayed = 1000
    
    nb_points_displayed = st.slider(
        'Number of points displayed',
        10, 1000, step=10, key='nb_points_displayed'
    )

    # Slider for gamma parameter
    if 'gamma' not in st.session_state:
        st.session_state.gamma = 0.5

    gamma = st.slider(
        'Discount factor',
        0.,1.,step=0.05, key = 'gamma', on_change = reinit_curve_data
    )


# Add a tab for data distribution and another for the benchmark
tab1, tab2 = st.tabs(['Data distribution', 'Learning curves'])

def plot_static_distributions():
    X_train, y_train, _, _ = generate_dataset(st.session_state.data_generator2D, st.session_state.size_train, 0)
    fig = px.scatter(
        x=X_train[:, 0], 
        y=X_train[:, 1],
        color= y_train,
        title='2D equivalent of the data distribution')
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    return fig

def plot_dynamic_distributions():
    data = []
    t = 0
    for i in range(st.session_state.nb_steps+1):
        X_train, y_train, _, _ = generate_dataset(st.session_state.data_generator2D, st.session_state.size_train, 0,t)
        df_data = pd.DataFrame(dict(
            t=round(t,2),
            x=X_train[:st.session_state.nb_points_displayed, 0],
            y=X_train[:st.session_state.nb_points_displayed, 1],
            label=y_train[:st.session_state.nb_points_displayed]
        ))
        data.append(df_data)
        t += st.session_state.dt

    df_distributions = pd.concat(data, axis=0)

    fig = px.scatter(
        df_distributions,
        x="x",
        y="y",
        color="label",
        animation_frame="t",
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 0
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = int(1000*st.session_state.dt)
    return fig


# Add a plot for the data distribution with plotly
with tab1:
    if st.session_state.data_generator.time_dependant:
        fig = plot_dynamic_distributions()
    else:
        fig = plot_static_distributions()
    st.plotly_chart(fig)

# Add a plot for the benchmark with plotly
with tab2:
    while_loop_entered = False
    while(len(st.session_state.all_accuracies_learner)<st.session_state.nb_replications_max):
        while_loop_entered = True
        learner = construct_learner(st.session_state.classifier, st.session_state.query_strategy)
        if st.session_state.data_generator.time_dependant:
            x = [i*dt for i in range(st.session_state.nb_steps+1)]
            accuracies_basic, accuracies_learner = compute_time_accuracies(st.session_state.data_generator, 
                                                                           st.session_state.size_train, 
                                                                           st.session_state.size_val, 
                                                                           st.session_state.basic_train, 
                                                                           st.session_state.nb_queries_per_step, 
                                                                           st.session_state.nb_steps, 
                                                                           learner, 
                                                                           st.session_state.classifier, 
                                                                           st.session_state.dt,
                                                                           st.session_state.gamma)
        else:
            X_train, y_train, X_val, y_val = generate_dataset(st.session_state.data_generator, st.session_state.size_train, st.session_state.size_val)
            x = [i for i in range(st.session_state.basic_train,
                            st.session_state.basic_train+st.session_state.nb_queries+1)]
            accuracies_basic, accuracies_learner = compute_accuracies(X_train,
                                                                    y_train, 
                                                                    X_val, 
                                                                    y_val, 
                                                                    st.session_state.basic_train, 
                                                                    st.session_state.nb_queries, 
                                                                    learner, 
                                                                    st.session_state.classifier)
        if len(st.session_state.all_accuracies_learner)==0:
            st.session_state.all_accuracies_learner = np.array([accuracies_learner])
            st.session_state.all_accuracies_basic = np.array([accuracies_basic])
        else:
            st.session_state.all_accuracies_learner = np.concatenate((st.session_state.all_accuracies_learner, np.array([accuracies_learner])),axis=0)
            st.session_state.all_accuracies_basic = np.concatenate((st.session_state.all_accuracies_basic, np.array([accuracies_basic])),axis=0)
        if st.session_state.data_generator.time_dependant:
            df = pd.DataFrame(dict(
                time=x,
                basic=np.mean(st.session_state.all_accuracies_basic,axis=0),
                learner=np.mean(st.session_state.all_accuracies_learner,axis=0)
            ))
            fig = px.line(df, 
                        x="time", 
                        y=["basic","learner"],
                        title=f'Accuracy of the classifier with basic training and active learning on {len(st.session_state.all_accuracies_learner)} replications')
        
        else:
            df = pd.DataFrame(dict(
                queries=x,
                basic=np.mean(st.session_state.all_accuracies_basic,axis=0),
                learner=np.mean(st.session_state.all_accuracies_learner,axis=0)
            ))
            fig = px.line(df, 
                        x="queries", 
                        y=["basic","learner"],
                        title=f'Accuracy of the classifier with basic training and active learning on {len(st.session_state.all_accuracies_learner)} replications')
        if "plot_accuracies" in st.session_state:
            st.session_state.plot_accuracies.empty()
        st.session_state.plot_accuracies = st.plotly_chart(fig)

    # if we do not enter the while loop we need to plot the graph !
    if(not while_loop_entered):
        if st.session_state.data_generator.time_dependant:
                    x = [i*dt for i in range(st.session_state.nb_steps+1)]
        else:
            x = [i for i in range(st.session_state.basic_train,
            st.session_state.basic_train+st.session_state.nb_queries+1)]
        df = pd.DataFrame(dict(
            queries=x,
            basic=np.mean(st.session_state.all_accuracies_basic[:st.session_state.nb_replications_max,:],axis=0),
            learner=np.mean(st.session_state.all_accuracies_learner[:st.session_state.nb_replications_max,:],axis=0)
        ))
        fig = px.line(df, 
                    x="queries", 
                    y=["basic","learner"],
                    title=f'Accuracy of the classifier with basic training and active learning on {nb_replications_max} replications')
        if "plot_accuracies" in st.session_state:
            st.session_state.plot_accuracies.empty()
        st.session_state.plot_accuracies = st.plotly_chart(fig)
import streamlit as st
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px


from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from generators.generator import GaussianGenerator, DataGenerators, not_convex
from learners.comparator import create_two_gaussians, construct_learner, generate_dataset, compute_accuracies, plot_accuracies, do_benchmark, create_SVC
from learners.time_dependancy import construct_water_level_data_generator, compute_time_accuracies

st.sidebar.title('Active learning dashboard')

data_generator_names = ['Two Gaussians', 'Not convex']
classifier_names = ['Logistic Regression', 'Naive Bayes', 'SVM']
query_strategy_names = ['Uncertainty sampling', 'Margin sampling', 'Entropy sampling']

# Functions to update session state
def update_data_generator():
    if st.session_state.data_generator_name == 'Two Gaussians':
        st.session_state.data_generator = create_two_gaussians(dim=st.session_state.dimension, first_dim_mean=1, first_dim_std=0.5, other_dim_stds=1)
    elif st.session_state.data_generator_name == 'Not convex':
        st.session_state.data_generator  = not_convex(dim=st.session_state.dimension)

def update_classifier():
    if st.session_state.classifier_name == 'Logistic Regression':
        st.session_state.classifier = LogisticRegression
    elif st.session_state.classifier_name == 'Naive Bayes':
        st.session_state.classifier = GaussianNB
    elif st.session_state.classifier_name == 'SVM':
        st.session_state.classifier = create_SVC

def update_query_strategy():
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
    # Slider for the dimension
    if 'dimension' not in st.session_state:
        st.session_state.dimension = 100

    dimension = st.slider(
        'Dimension',
        0, 300, step=10, 
        key='dimension',
        on_change=update_data_generator
    )

    # Selectbox for the data generator
    if 'data_generator_name' not in st.session_state:
        st.session_state.data_generator_name = 'Two Gaussians'
    if 'data_generator' not in st.session_state:
        st.session_state.data_generator = create_two_gaussians(dim=st.session_state.dimension, first_dim_mean=1, first_dim_std=0.5, other_dim_stds=1)

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
        0, 3000, step=100, key='size_train'
    )

    # Slider for the size of the validation set with initial value 1000
    if 'size_val' not in st.session_state:
        st.session_state.size_val = 1000

    size_val = st.slider(
        'Size of the validation set',
        0, 3000, step=100, key='size_val'
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
        0, 100, step=1, key='basic_train'
    )

    # Slider for the number of queries with initial value 290
    if 'nb_queries' not in st.session_state:
        st.session_state.nb_queries = 290

    nb_queries = st.slider(
        'Number of queries',
        0, 1000, step=10, key='nb_queries'
    )

with sbtab4:
    # Radio button to indicate if dataset is static or dynamic
    if 'is_static' not in st.session_state:
        st.session_state.is_static = True
    
    is_static = st.radio(
        'Dataset i static',
        (True, False),
        key='is_static'
    )


    # Slider for the number of steps
    if 'nb_steps' not in st.session_state:
        st.session_state.nb_steps = 200
    
    nb_steps = st.slider(
        'Number of steps',
        0, 1000, step=10, key='nb_steps'
    )

    # Slider for the number of queries per step
    if 'nb_queries_per_step' not in st.session_state:
        st.session_state.nb_queries_per_step = 5
    
    nb_queries_per_step = st.slider(
        'Number of queries per step',
        0, 100, step=1, key='nb_queries_per_step'
    )

    # Slider for dt
    if 'dt' not in st.session_state:
        st.session_state.dt = 0.1
    
    dt = st.slider(
        'dt',
        0.0, 1.0, step=0.1, key='dt'
    )

    # Slider for number of points displayed
    if 'nb_points_displayed' not in st.session_state:
        st.session_state.nb_points_displayed = 1000
    
    nb_points_displayed = st.slider(
        'Number of points displayed',
        0, 1000, step=10, key='nb_points_displayed'
    )
    

# Add a tab for data distribution and another for the benchmark
tab1, tab2 = st.tabs(['Data distribution', 'Learning curves'])

def plot_static_distributions():
    X_train, y_train, X_val, y_val = generate_dataset(st.session_state.data_generator, st.session_state.size_train, st.session_state.size_val)
    fig = px.scatter(
        x=X_train[:, 0], 
        y=X_train[:, 1],
        color= y_train,
        title='First and second dimensions of data distribution')
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    return fig

def plot_dynamic_distributions():
    data_gen = construct_water_level_data_generator(st.session_state.dimension)

    data = []
    t = 0
    for i in range(st.session_state.nb_steps+1):
        X_train, y_train, X_val, y_val = generate_dataset(data_gen, st.session_state.size_train, st.session_state.size_val,t)
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
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 0
    return fig


# Add a plot for the data distribution with plotly
with tab1:
    if st.session_state.is_static:
        fig = plot_static_distributions()
    else:
        fig = plot_dynamic_distributions()
    st.plotly_chart(fig)

# Add a plot for the benchmark with plotly
with tab2:
    X_train, y_train, X_val, y_val = generate_dataset(st.session_state.data_generator, st.session_state.size_train, st.session_state.size_val)
    learner = construct_learner(st.session_state.classifier, st.session_state.query_strategy)
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
    df = pd.DataFrame(dict(
        queries=x,
        basic=accuracies_basic,
        learner=accuracies_learner
    ))
    fig = px.line(df, 
                  x="queries", 
                  y=["basic","learner"],
                  title='Accuracy of the classifier with basic training and active learning')
    st.plotly_chart(fig)

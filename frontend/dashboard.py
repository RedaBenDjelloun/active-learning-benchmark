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
tabs = ['Dataset', 'Classifier', 'Learner']
sbtab1, sbtab2, sbtab3 = st.sidebar.tabs(tabs)


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

# Add a tab for data distribution and another for the benchmark
tab1, tab2 = st.tabs(['Data distribution', 'Learning curves'])



X_train, y_train, X_val, y_val = generate_dataset(st.session_state.data_generator, st.session_state.size_train, st.session_state.size_val)
learner = construct_learner(st.session_state.classifier, st.session_state.query_strategy)

# Add a plot for the data distribution with plotly
with tab1:
    fig = px.scatter(
        x=X_train[:, 0], 
        y=X_train[:, 1],
        color= y_train,
        title='First and second dimensions of data distribution')
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    st.plotly_chart(fig)

# Add a plot for the benchmark with plotly
with tab2:
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

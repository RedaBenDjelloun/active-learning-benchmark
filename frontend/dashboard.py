import streamlit as st
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from generators.generator import GaussianGenerator, DataGenerators, not_convex
from learners.comparator import create_two_gaussians, construct_learner, generate_dataset, compute_accuracies, plot_accuracies, do_benchmark, create_SVC

data_generator_names = ['Two Gaussians', 'Not convex']

st.sidebar.title('Active learning dashboard')

# Slider for the dimension
if 'dimension' not in st.session_state:
    st.session_state.dimension = 100
    
st.session_state.dimension = st.sidebar.slider(
    'Dimension',
    0, 300, value=100, step=10
)

# Selectbox for the data generator
if 'data_generator' not in st.session_state:
    st.session_state.data_generator = create_two_gaussians(dim=st.session_state.dimension, first_dim_mean=1, first_dim_std=0.5, other_dim_stds=1)

select_data_generator = st.sidebar.selectbox(
    'Data generator',
    data_generator_names
)
if select_data_generator == 'Two Gaussians':
    st.session_state.data_generator = create_two_gaussians(dim=st.session_state.dimension, first_dim_mean=1, first_dim_std=0.5, other_dim_stds=1)
elif select_data_generator == 'Not convex':
    st.session_state.data_generator  = not_convex(dim=st.session_state.dimension)

# Selectbox for the classifier
if 'classifier' not in st.session_state:
    st.session_state.classifier = LogisticRegression

classifier_names = ['Logistic Regression', 'Naive Bayes', 'SVM']
select_classifier = st.sidebar.selectbox(
    'Classifier',
    classifier_names
)
if select_classifier == 'Logistic Regression':
    st.session_state.classifier = LogisticRegression
elif select_classifier == 'Naive Bayes':
    st.session_state.classifier = GaussianNB
elif select_classifier == 'SVM':
    st.session_state.classifier = create_SVC

# Selectbox for the query strategy
if 'query_strategy' not in st.session_state:
    st.session_state.query_strategy = uncertainty_sampling

query_strategy_names = ['Uncertainty sampling', 'Margin sampling', 'Entropy sampling']
select_query_strategy = st.sidebar.selectbox(
    'Query strategy',
    query_strategy_names
)
if select_query_strategy == 'Uncertainty sampling':
    st.session_state.query_strategy = uncertainty_sampling
elif select_query_strategy == 'Margin sampling':
    st.session_state.query_strategy = margin_sampling
elif select_query_strategy == 'Entropy sampling':
    st.session_state.query_strategy = entropy_sampling

# Slider for the number of basic train with initial value 10
if 'basic_train' not in st.session_state:
    st.session_state.basic_train = 10

st.session_state.basic_train = st.sidebar.slider(
    'Number of basic train',
    0, 100, value=10, step=1
)

# Slider for the number of queries with initial value 290
if 'nb_queries' not in st.session_state:
    st.session_state.nb_queries = 290

st.session_state.nb_queries = st.sidebar.slider(
    'Number of queries',
    0, 1000, value=290, step=10
)

# Slider for the size of the training set with initial value 1000
if 'size_train' not in st.session_state:
    st.session_state.size_train = 1000

st.session_state.size_train = st.sidebar.slider(
    'Size of the training set',
    0, 10000, value=1000, step=100
)

# Slider for the size of the validation set with initial value 1000
if 'size_val' not in st.session_state:
    st.session_state.size_val = 1000

st.session_state.size_val = st.sidebar.slider(
    'Size of the validation set',
    0, 10000, value=1000, step=100
)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
do_benchmark(ax, st.session_state.data_generator, st.session_state.classifier, st.session_state.query_strategy, st.session_state.basic_train, st.session_state.nb_queries, st.session_state.size_train, st.session_state.size_val)
st.pyplot(fig)

# left_column, right_column = st.columns(2)
# # You can use a column just like st.sidebar:
# left_column.button('Press me!')

# # Or even better, call Streamlit functions inside a "with" block:
# with right_column:
#     chosen = st.radio(
#         'Sorting hat',
#         ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
#     st.write(f"You are in {chosen} house!")


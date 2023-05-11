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

# Slider for the dimension
dimension = st.sidebar.slider(
    'Dimension',
    0, 300, value=100, step=10
)

# Selectbox for the data generator
select_data_generator = st.sidebar.selectbox(
    'Data generator',
    data_generator_names
)
if select_data_generator == 'Two Gaussians':
    data_generator = create_two_gaussians(dim=dimension, first_dim_mean=1, first_dim_std=0.5, other_dim_stds=1)
elif select_data_generator == 'Not convex':
    data_generator  = not_convex(dim=dimension)

# Selectbox for the classifier
classifier_names = ['Logistic Regression', 'Naive Bayes', 'SVM']
select_classifier = st.sidebar.selectbox(
    'Classifier',
    classifier_names
)
if select_classifier == 'Logistic Regression':
    classifier = LogisticRegression
elif select_classifier == 'Naive Bayes':
    classifier = GaussianNB
elif select_classifier == 'SVM':
    classifier = create_SVC

# Selectbox for the query strategy
query_strategy_names = ['Uncertainty sampling', 'Margin sampling', 'Entropy sampling']
select_query_strategy = st.sidebar.selectbox(
    'Query strategy',
    query_strategy_names
)
if select_query_strategy == 'Uncertainty sampling':
    query_strategy = uncertainty_sampling
elif select_query_strategy == 'Margin sampling':
    query_strategy = margin_sampling
elif select_query_strategy == 'Entropy sampling':
    query_strategy = entropy_sampling

# Slider for the number of basic train with initial value 10
basic_train = st.sidebar.slider(
    'Number of basic train',
    0, 100, value=10, step=1
)

# Slider for the number of queries with initial value 290
nb_queries = st.sidebar.slider(
    'Number of queries',
    0, 1000, value=290, step=10
)

# Slider for the size of the training set with initial value 1000
size_train = st.sidebar.slider(
    'Size of the training set',
    0, 10000, value=1000, step=100
)

# Slider for the size of the validation set with initial value 1000
size_val = st.sidebar.slider(
    'Size of the validation set',
    0, 10000, value=1000, step=100
)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
do_benchmark(ax, data_generator, classifier, query_strategy, basic_train, nb_queries, size_train, size_val)
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


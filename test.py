from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os


df = pd.read_csv('dataset.csv')
setup(df, target='Salary', silent=True)
setup_df = pull()
print(setup_df)
best_model = compare_models()
compare_df = pull()
print(compare_df)
save = save_model(best_model, 'best_model')
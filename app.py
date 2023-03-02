from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 

st.title("AutoML APP")
st.header("Welcome")




save = None
side = st.sidebar

with side:
    #st.image('https://www.justintodata.com/wp-content/uploads/2022/03/h2o-automl-python.png')
    st.header('Get Started here')
    st.info("Upload your csv to get started")
    file = st.file_uploader("Upload Your Dataset")


if file == None:
    st.write("This is a quick AutoML app best suited to smaller datasets.")
    st.write("To get started, use the side bar to the left.")
    st.write("You can make you model and download the best one.")
    st.text("Steps:\n1.Upload\n2.Describe\n3.Model\n4.Download")

if file:
    df = pd.read_csv(file, index_col=None)
    df.to_csv('dataset.csv', index=None)
    with st.expander("view data"):
        st.dataframe(df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.dataframe(df.describe())
    with col2:
        st.subheader("The data has been loaded")
        "You can proceed to generate a pandas profile report and/or model building."


    with st.expander("Do Pandas Profiling"):
        st.subheader("Exploratory Data Analysis")
        profile_df = df.profile_report()
        prf = st_profile_report(profile_df)

with st.expander("Model"):
    if file:
        st.table([i for i in df.columns if type('df'+str(i)) == str])
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Modelling'):
            setup(df, target=chosen_target, silent=False, use_gpu=True)
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save = save_model(best_model, 'best_model')
if save:
    if st.button("Download"):
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")




















page_bg_img = '''

<style>
body {
background-image: url("https://www.automl.org/wp-content/uploads/slider2/AutoML_WasistAutoML_slider.png");
background-size: cover; opacity:0.8;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)
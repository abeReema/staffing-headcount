import streamlit as st

css_path = 'styles.txt'


with open(css_path, 'r') as file:
    css = file.read() 

st.set_page_config(
    page_title="Home",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.success("Select View Above")

st.markdown(css, unsafe_allow_html=True)

st.markdown('<div class="page-title">Staffing Model</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">one-stop shop for hiring</div>', unsafe_allow_html=True)
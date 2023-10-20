import streamlit as st
import time
from utils import *
st.set_page_config(
    page_title="Image Regression Demo",
    layout="centered",
    menu_items={},
    initial_sidebar_state="collapsed"
)


st.title("Image Regression Demo")

st.markdown('''
Checkout [About](/About) for details on how to use this demo.
            
Tip: Smaller images load and run faster than higher resolution ones
''')

with st.form("My-Form"):
    image_file = st.file_uploader("Upload an image", type = ["png", "jpeg", "jpg"])
    col1, col2 = st.columns(2)
    with col1:
        network_type = st.selectbox(
            label = "Network Type",
            options = ["Vanilla NN", "Fourier Feature Network"],
            index = 1
        )
        
        lr = st.number_input(
            label="lr (Learning Rate)", 
            min_value=0.00001, value = 0.003, format="%0.4f", step = 0.0001,
            help="Learning Rate")
        
        gaussian_mapping_size = None
        if network_type == "Fourier Feature Network":
            gaussian_mapping_size = st.number_input(
                label="gaussian mapping size",
                min_value=128, step = 1, value=256, max_value=1024,
                help = "Length of the Gaussian Normal Batrix B"
            )
    
        iters = st.number_input(
            label = "training iters", min_value=30, max_value=10000,value=500,
            help="No. of iters to train the model")
        
    with col2:
        n_hidden = st.number_input(
            label="n_hidden", 
            min_value=64,step=1,value = 128, 
            help="No. of neurons in the hidden layers")
        
        gamma = st.number_input(
            label = "gamma(lr decay constant)", 
            min_value=0.5, value=0.995, max_value=1.000, format="%0.4f", step=0.001,
            help = "Rate at which the learning rate decays")
        
        gaussian_scale_factor = None
        if network_type == "Fourier Feature Network":
            gaussian_scale_factor = st.number_input(
                label="gaussian scale factor",
                min_value=1,max_value=1000, step = 1, value = 5,
                help = "Standard Deviation Value for the Gaussian Normal Matrix"
            )

        

    submitted = st.form_submit_button("Run")

    if submitted:
        if image_file is None:
            st.error("Upload an image first")
        else:
            model_type = "VanillaNN" if network_type == "Vanilla NN" else "FFN"
            setup(
                image_file,
                model_type, 
                n_hidden, lr, gamma, 
                gaussian_mapping_size, gaussian_scale_factor
            )
            
            success = st.success("Setup Complete")
            with st.spinner("Starting Training"):
                time.sleep(1)
            success.empty()
            
            train(iters)
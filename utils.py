import io
from PIL import Image
import streamlit as st

import torch
from torch import nn
import torchvision.transforms as transforms

from model import *

def setup(
        image_file, 
        model_type:str,        
        n_hidden:int, 
        lr:float, 
        gamma:float,
        gaussian_mapping_size:int = None,
        gaussian_scale_factor:int = None
    )->None:
    '''
    Creates and lLoads necessary variables into the session state

    ### Params:
    - image_file : File Object returned by the ```st.file_uploader```
    - model_type (str) : Specifies the type of Model Architecture to use
    - n_hidden (int) : No of neurons in the hidden layers of the neural network
    - lr (float) : The learning rate for the training algorithm
    - gamma (float) : Specifies the rate at which the learning rate decays
                      Used by ```torch.optim.lr_scheduler.ExponentialLR```
    - gaussian_mapping_size (int) : Number of rows in Gaussian Matrix B
    - gaussian_scale_factor (int) : Std of B

    ### Returns
    (None) 
    '''

    bytes = image_file.getvalue()
    image = Image.open(io.BytesIO(bytes))

    img_tensor = torch.permute(transforms.ToTensor()(image), (1, 2, 0))
    
    # drop 4th channel (alpha) if present, especially in png images
    img_tensor = img_tensor[:, :, :3]

    im_height = img_tensor.shape[0]    # image height
    im_width = img_tensor.shape[1]    # image width
    
    st.session_state.im_height = im_height
    st.session_state.im_width = im_width

    st.session_state.target_image = img_tensor.reshape(
        (im_height, im_width, 3)
    ).cpu().detach().numpy()

    # Normalized (-1 to +1) Image Coordinates
    with st.spinner("Building X"):
        x = [
                [
                    [
                        (i - (im_height - 1)/2)/(im_height - 1), 
                        (j - (im_width - 1)/2)/(im_width - 1)
                    ] for j in range(im_width)
                ] for i in range(im_height)
        ]
        st.session_state.x = torch.tensor(x).view(im_height*im_width, 2)*2

    # Target Values (Real Image)
    with st.spinner("Building Y"):
        y = img_tensor.detach().clone()
        st.session_state.y = y.view(im_height * im_width, 3)

    with st.spinner("Building Model"):
        model_objects = build_model(
            model_type,
            n_hidden, lr, gamma, 
            gaussian_mapping_size,gaussian_scale_factor
        )
        st.session_state.model = model_objects[0]
        st.session_state.loss_fn = model_objects[1]
        st.session_state.optimizer = model_objects[2]
        st.session_state.scheduler = model_objects[3]
        
        total_trainable_params = sum(
            p.numel() for p in model_objects[0].parameters() if p.requires_grad
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Model Built")
            st.empty()
        with col2:
            st.metric(label = "Total number of trainable parameters",value=total_trainable_params)
            st.empty()
    # Will be filled while training the model
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.img_holder1 = st.empty()
        st.session_state.img_title_holder1 = st.empty()
    with col2:
        st.session_state.img_holder2 = st.empty()
        st.session_state.img_title_holder2 = st.empty()


def build_model(
        model_type:str,
        n_hidden:int, lr:float,gamma:float, 
        gaussian_mapping_size:int = None,
        gaussian_scale_factor:int = None
    ):
    '''
    Builds and returns the necessary model and optimization objects

    ### Params:
    - model_type (str) : The type of the Model Architecture to use
    - n_hidden (int) : No of neurons in the hidden layers of the neural network
    - lr (float) : The learning rate for the training algorithm
    - gamma (float) : Specifies the rate at which the learning rate decays
                      Used by ```torch.optim.lr_scheduler.ExponentialLR```
    - gaussian_mapping_size (int) : Number of rows in Gaussian Matrix B
    - gaussian_scale_factor (int) : Std of B

    ### Returns
    (model, loss_fn, optimizer, scheduler) object tuple
    '''
    
    if model_type == "VanillaNN":
        model = VanillaNN(n_hidden)

    elif model_type == "FFN":
        model = FFN(n_hidden, gaussian_mapping_size, gaussian_scale_factor)
    else:
        st.error("Invalid Model Type")
        st.stop()
        return
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma)
    return model, loss_fn, optimizer, scheduler


def update()->None:
    '''
    Updates the Ground Truth and Learned Image Previews on screen 
    during the training process

    ### Params:
    (None)

    ### Returns :
    (None)
    '''
    st.session_state.model.eval()

    im_height = st.session_state.im_height
    im_width = st.session_state.im_width

    img_tensor_new = st.session_state.model(st.session_state.x)
    img_tensor_new = img_tensor_new.squeeze().reshape(im_height, im_width, 3)

    st.session_state.img_title_holder1.write("Original Image")
    st.session_state.img_holder1.image(st.session_state.target_image)

    st.session_state.img_title_holder2.write("Learned Image")
    st.session_state.img_holder2.image(img_tensor_new.cpu().detach().numpy())

    st.session_state.model.train()

def train(num_iterations:int = 2000)->None:
    '''
    Commences the model training process

    ### Params:
    - num_iterations (int) : number of iterations to train the model

    ### Returns
    (None)
    '''
    st.session_state.model.train()
    pbar = st.progress(value=0)
    col1, col2 = st.columns(2)
    with col1:
        metric1 = st.empty()
    with col2:
        metric2 = st.empty()

    precision_scale = 10**7
    format_val = lambda i: (int(i * precision_scale))/precision_scale

    for iteration in range(num_iterations):
        y_hat = st.session_state.model(st.session_state.x)       
        loss = st.session_state.loss_fn(y_hat, st.session_state.y)
                
        loss.backward()
        st.session_state.optimizer.step()
        st.session_state.optimizer.zero_grad()
        st.session_state.scheduler.step()

        pbar.progress(value = (iteration+1)/num_iterations, text=f'''
                        {iteration}/{num_iterations} iterations Complete''')
        
        metric1.metric(
            label = "Learning Rate",
            value = format_val(st.session_state.optimizer.param_groups[0]['lr'])
        )

        metric2.metric(
            label = "Loss Value",
            value = format_val(loss.item())
        )
        if iteration % 5 == 0:
            update()
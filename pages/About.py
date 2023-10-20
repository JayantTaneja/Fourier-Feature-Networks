import streamlit as st

st.set_page_config(
    page_title="About",
    layout="centered",
    menu_items={},
)


st.title("About")


st.markdown('''

This interactive demo aims to preovide an interface for playing around with 
basic Fourier Feature Networks 
[link to the original paper](https://arxiv.org/abs/2006.10739) and compare
how fast they are compared to Vanilla neural networks

---
''')

video_file = open("video.mp4", "rb")
video_bytes = video_file.read()
st.video(video_bytes)

st.markdown('''
### Hyperparameter Details
This section describes the various hyperparameters and settings that you
can tweak in the demo :
            
- ```Network Type```:
    - Specifies the type of Neural Network architecture to be used

- ```n_hidden```:
    - Specifies the number of neurons in the hidden layers

- ```lr```:
    - Specifies the learning rate at which the model learns
    - A higher learning rate may mean the model learns faster but will
    cause the training process to be quite unstable
    - A lower learning rate will cause the model to learn in a more stable
    but slower manner

- ```gamma```:
    - Specifies the rate at which the learning rate decays
    - Uses exponential decay to slowly decrease the learning rate
    - See [link](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html)

- ```gaussian mapping size```:
    - Specifies the mapping size of the Gaussian Normal Matrix ```B```
    as mentioned in the original paper

- ```gaussian scale factor```:
    - Specifies the standard deviation of the Gaussian Normal Matrix ```B```

- ```training iters```:
    - Specifies the number of iterations for the model to train
''')
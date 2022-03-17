import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
import tezhenggx as tz


class Act_fun(nn.Module):
    def __init__(self):
        super(Act_fun, self).__init__()

    def forward(self, x):
        x = torch.sigmoid(x)
        x = x * 100
        return x

st.title('demo')
add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)

# 载入数据
@st.cache
def load():
    X_train = pd.read_csv(r'C:\Users\zhao\Desktop\streamlit_webapp\X_train_sc_st_pca.csv')
    X_test = pd.read_csv(r'C:\Users\zhao\Desktop\streamlit_webapp\X_test_sc_st_pca.csv')
    y_train = pd.read_csv(r'C:\Users\zhao\Desktop\streamlit_webapp\train_y.csv')
    y_test = pd.read_csv(r'C:\Users\zhao\Desktop\streamlit_webapp\test_y.csv')
    test_data = pd.read_csv(r'D:\jupyter_project\大创\test_data.csv')
    train_data = pd.read_csv(r'D:\jupyter_project\大创\train_data.csv')
    X_train = np.array(X_train)
    X_test=np.array(X_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    return X_train,X_test,y_train,y_test,train_data,test_data


data_load_state = st.text('Loading data...')
X_train,X_test,y_train,y_test,train_data,test_data = load()
data_load_state.text('Loading data...done!')
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(train_data)
lr_to_filter = st.slider('学习率X10^5', 1,50,10)
num_to_filter = st.slider('学习次数', 1,5000,1000)
#模型
input_size = X_train.shape[1]
hidden_size1 = 100
output_size = 1
batch_size = 10
# 框架

my_nn = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size1),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size1, output_size),
    Act_fun(),
)
cost = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(my_nn.parameters(), lr=lr_to_filter*10**-4)
for i in range(num_to_filter):
    batch_loss = []
    # MINI-Batch方法来进行训练
    for start in range(0, len(X_train), batch_size):
        end = start + batch_size if start + batch_size < len(X_train) else len(X_train)
        xx = torch.tensor(X_train[start:end], dtype=torch.float, requires_grad=True)
        yy = torch.tensor(y_train[start:end], dtype=torch.float, requires_grad=True)
        prediction = my_nn(xx)
        # 降为一维，否则loss会告警
        prediction = prediction.squeeze(-1)
        loss = cost(prediction, yy)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        batch_loss.append(loss.data.numpy())

    # 打印损失
    if i % 100 == 0:
        #losses.append(np.mean(batch_loss))
        st.write('第',i,'次训练', '均方误差为:',np.mean(batch_loss))


import streamlit as st
import numpy as np
from load_data import load_all
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from Input_preprocess import Input_preprocess



def run(Input_data=[0,0,0,0,0,0,0,0,'Silica_sand','bubbling fluidized bed'],Modle = 0):
    st.title('ANN')
    class Act_fun(nn.Module):
        def __init__(self):
            super(Act_fun, self).__init__()

        def forward(self, x):
            x = torch.sigmoid(x)
            x = x * 100
            return x



    # 载入数据
    target = st.sidebar.selectbox("选择要预测的气体", (
    "CO", "H2", "CH4", "CO2" ))
    data_load_state = st.text('Loading data...')
    X_train, X_test, y_train, y_test, train_data, test_data = load_all(target)
    data_load_state.text('Loading data...done!')
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(train_data)
    lr_to_filter = st.slider('学习率X10^5', 1, 100, 10)
    num_to_filter = st.slider('学习次数', 1, 500, 100)

    # 模型
    input_size = X_train.shape[1]
    hidden_size1 = 20
    output_size = 1
    batch_size = 10
    # 框架
    my_nn = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size1),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size1, output_size),
        Act_fun(),
    )
    st.subheader("打印损失值")
    cost = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(my_nn.parameters(), lr=lr_to_filter * 10 ** -5)
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
        if i % 50 == 0:
            # losses.append(np.mean(batch_loss))
            st.write('第', i, '次训练', 'loss:', np.mean(batch_loss))

    st.subheader("评价指标")
    train_predict = my_nn(torch.tensor(X_train, dtype=torch.float)).data.numpy().reshape(-1)
    train_rmse = (np.mean((y_train.reshape(-1) - train_predict) ** 2))**0.5
    train_mape = np.mean(abs((y_train.reshape(-1) - train_predict)/y_train.reshape(-1))*100/y_train.shape[1]) # 平均绝对百分误差
    st.write('训练集均方根误差：', train_rmse,'训练集平均绝对百分误差：',train_mape,"%")

    test_predict = my_nn(torch.tensor(X_test, dtype=torch.float)).data.numpy().reshape(-1)
    test_mape = np.mean(abs((y_test.reshape(-1) - test_predict)/y_test.reshape(-1)) * 100 / y_test.shape[1])  # 平均绝对百分误差
    test_rmse = (np.mean((y_test.reshape(-1) - test_predict) ** 2))**0.5
    st.write('测试集均方根误差：',test_rmse, '测试集平均绝对百分误差：',test_mape,"%")
#     if Input_data != [0,0,0,0,0,0,0,0,'Silica_sand','bubbling fluidized bed']:
#         #  根据输入数据预测
#         input_predict = my_nn(torch.tensor(Input_preprocess(Input_data), dtype=torch.float)).data.numpy().reshape(-1)
#         st.subheader('**预测值**')
#         st.write(target,'[%vol_N2_free]=',input_predict[0])
#     else:
#         st.subheader('**请在侧边栏上传文件或输入数据**')
    
    
    import pandas as pd
    if Modle == 0:
        st.subheader("请在侧边栏输入反应参数")
    elif Modle == 1:
        input_predict = my_nn(torch.tensor(Input_preprocess(Input_data), dtype=torch.float)).data.numpy().reshape(-1)
        st.subheader('气体产出预测为：')
        st.write(target, '[%vol_N2_free]=', input_predict[0])
    elif Modle ==-1:
        st.subheader("气体产出预测为：")
        load_state = st.text('Loading...')
        input_data = Input_preprocess(Input_data)
        l = []
        #st.write(input_data)
        input_data = Input_preprocess(Input_data)
        input_predict = my_nn(torch.tensor(Input_preprocess(Input_data), dtype=torch.float)).data.numpy().reshape(-1)
        l.append(input_predict)
        l = pd.DataFrame(l).T
        dic = {target+"[%vol_N2_free]":l[0]}
        df = pd.DataFrame(dic)
        st.write(df)
        load_state.text("loading...done")




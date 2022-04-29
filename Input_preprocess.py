# 导入包
import numpy as np
import pandas as pd
from load_data import load_all,load_csv
import streamlit as st

# 馊主意挺多了，以后肯定会优化
@st.cache
def Input_preprocess(Input_data):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    #  独热编码
    Input_data = pd.DataFrame(Input_data)
    c = ['GY', 'CH4', 'CO2', 'CO', 'H2']
    train_data = load_csv('train_data.csv')
    test_data = load_csv('test_data.csv')
    data = pd.concat([train_data, test_data])
    #print(test_data)
    X_data = data.drop(c, axis=1)
    features_columns = [col for col in X_data.columns]
    Input_data.columns = features_columns
    #  插进去，一起独热编码
    X_data = pd.concat([X_data,Input_data])
    X_data = pd.get_dummies(X_data)
    #再拆出来
    input_data = X_data[-len(Input_data):]



    #  跳过去除异常值，开始下面的步骤
    data = load_csv("x_data_no_outlier.csv")
    l1 = int(0.8 * data.shape[0])
    l2 = data.shape[0] - l1
    train_data = data.head(l1)
    test_data = data.tail(l2)
    test_data = pd.concat([test_data,input_data])

    #  归一化
    from sklearn import preprocessing

    features_columns = [col for col in train_data.columns]
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler = min_max_scaler.fit(train_data)

    train_data_scaler = min_max_scaler.transform(train_data)
    test_data_scaler = min_max_scaler.transform(test_data)
    #print("21yngfo hngiebgp1g")
    train_data_scaler = pd.DataFrame(train_data_scaler)
    train_data_scaler.columns = features_columns

    test_data_scaler = pd.DataFrame(test_data_scaler)
    test_data_scaler.columns = features_columns

    # 标准化
    from sklearn.preprocessing import PowerTransformer
    pt = PowerTransformer()  # 默认Yeo-johnson
    pt.fit(train_data_scaler)  # 同样以训练集创建
    train_data_sc_st = pt.transform(train_data_scaler)
    test_data_sc_st = pt.transform(test_data_scaler)

    train_data_sc_st = pd.DataFrame(train_data_sc_st)
    train_data_sc_st.columns = features_columns

    test_data_sc_st = pd.DataFrame(test_data_sc_st)
    test_data_sc_st.columns = features_columns


    # pca降维
    from sklearn.decomposition import PCA  # 主成分分析法

    X_train_sc_st = train_data_sc_st
    X_test_sc_st = test_data_sc_st

    # 保持99%的信息
    pca = PCA(n_components=0.95)
    X_train_sc_st_pca = pca.fit_transform(X_train_sc_st)
    X_test_sc_st_pca = pca.transform(X_test_sc_st)
    X_train_sc_st_pca = pd.DataFrame(X_train_sc_st_pca)
    X_test_sc_st_pca = pd.DataFrame(X_test_sc_st_pca)


    #  拆出来，输出
    input_data = np.array(X_test_sc_st_pca[-len(input_data):])
    print(input_data.shape)

    return input_data






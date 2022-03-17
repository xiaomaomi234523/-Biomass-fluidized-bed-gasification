import pandas as pd
import numpy as np
import streamlit as st
import os,inspect

# data里放那么多文件，用空间换时间了属于是
@st.cache
def load_all(target):
    # 获取绝对地址前缀（不知道为什么用不了相对地址）
    current_path=inspect.getfile(inspect.currentframe())
    dir_name=os.path.dirname(current_path)
    file_abs_path=os.path.abspath(dir_name)
   # st.write (file_abs_path)

    X_train = pd.read_csv(file_abs_path+r"\data\X_train_sc_st_pca_"+target+".csv")
    X_test = pd.read_csv(file_abs_path+r"\data\X_test_sc_st_pca_"+target+".csv")
    y_train = pd.read_csv(file_abs_path+r"\data\train_y_"+target+".csv")
    y_test = pd.read_csv(file_abs_path+r"\data\test_y_"+target+".csv")
    test_data = pd.read_csv(file_abs_path+r"\data\test_data.csv")
    train_data = pd.read_csv(file_abs_path+r"\data\train_data.csv")
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test, train_data, test_data

@st.cache
def load_csv(file_name):
     # 获取绝对地址前缀（不知道为什么用不了相对地址）
     current_path = inspect.getfile(inspect.currentframe())
     dir_name = os.path.dirname(current_path)
     file_abs_path = os.path.abspath(dir_name)

     file = pd.read_csv(file_abs_path+"\\data\\"+file_name)
     return file




# def to_Input(Input_data):
#     # 获取绝对地址前缀（不知道为什么用不了相对地址）
#     current_path = inspect.getfile(inspect.currentframe())
#     dir_name = os.path.dirname(current_path)
#     file_abs_path = os.path.abspath(dir_name)
#
#     df = pd.DataFrame(Input_data)
#     df.to_csv(file_abs_path+"\data\Input_data.csv", index=False, header=False)

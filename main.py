import streamlit as st
import pandas as pd
import numpy as np
from apps import GBDT_app,XGB_app, Adaboost_app, ElasticNet_app, ANN_app, RandomForest_app, SVR_app, home_app, \
    Lasso_app,Ridge_app



Input_data=[0,0,0,0,0,0,0,0,'Silica_sand','bubbling fluidized bed']
#Input_data = pd.DataFrame([Input_data])
Modle = 0 
list_a = np.arange(len(Input_data))
# 输入控件
st.sidebar.title("请输入反应参数")
input_mode = st.sidebar.selectbox("",("上传数据文件（注意格式）","单组数据"))
if input_mode == "上传数据文件（注意格式）":
    
    uploaded_file = st.sidebar.file_uploader("上传一个csv文件")
    if uploaded_file is not None:
        Input_data = pd.read_csv(uploaded_file,sep=',',usecols=list_a)
        Modle = -1
        #st.write(Input_data)
elif input_mode == "单组数据":
    Modle = 1
    C = st.sidebar.slider('C[wt.%dry basis]',0,100,50)
    H = st.sidebar.slider('H[wt.%dry basis]',0,100,6)
    O = st.sidebar.slider('O[wt.%dry basis]',0,100,44)
    if C+H+O >103 or C+H+O<95:
        st.sidebar.title("警告：您的输入错误，请重新输入,保证输入的元素和在[95,100]%内")
    else:
        Moisture = st.sidebar.slider('Moisture[%wt]',0,100,23)
        Ash = st.sidebar.slider('Ash[wt.%dry basis]',0,10,1)
        ER = 0.01*st.sidebar.slider('ER[-]*10^2',0,100,37)
        T = st.sidebar.slider('T[ºC]',500,1500,800)
        SB = 0.01*st.sidebar.slider('Steam/Biomass*10^2',0,100,0)
        Bed_material = st.sidebar.selectbox("",
            ('Silica sand','Alumina' , 'Olivine','Ofite','310S'))
        Bed_type = st.sidebar.selectbox("",
            ('fluidized bed','bubbling fluidized bed','atmospheric fluidized bed','circulating fluidized bed'))
        Input_data = [C,H,O,Moisture,Ash,ER,T,SB,Bed_material,Bed_type]
        Input_data = pd.DataFrame([Input_data])


# Once we have the dependencies, add a selector for the app mode on the sidebar.
st.sidebar.title("选择页面")
app_mode = st.sidebar.selectbox("",("home","Lasso","SVR","ANN","XGboost","Ridge", "ElasticNet","GBDT","RandomForest","Adaboost"))
if app_mode == "home":
    home_app.run(Input_data=Input_data,Modle = Modle)
    st.sidebar.success('请选择模型并试着调参吧!".')
elif app_mode == "ANN":
    ANN_app.run(Input_data=Input_data,Modle = Modle)
elif app_mode == "Lasso":
    Lasso_app.run(Input_data=Input_data,Modle = Modle)
elif app_mode == "ElasticNet":
    ElasticNet_app.run(Input_data=Input_data,Modle = Modle)
elif app_mode == "SVR":
    SVR_app.run(Input_data=Input_data,Modle = Modle)
elif app_mode == "GBDT":
    GBDT_app.run(Input_data=Input_data,Modle = Modle)
elif app_mode == "XGboost":
    XGB_app.run(Input_data=Input_data,Modle = Modle)
elif app_mode == "RandomForest":
    RandomForest_app.run(Input_data=Input_data,Modle = Modle)
elif app_mode == "Adaboost":
    Adaboost_app.run(Input_data=Input_data,Modle = Modle)
elif app_mode == "Ridge":
    Ridge_app.run(Input_data=Input_data,Modle = Modle)


for i in range(20):
    print("加载成功")




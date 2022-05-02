import copy
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os,inspect
import plotly as py
import plotly.graph_objs as go
import plotly.express as px 
from Input_preprocess import Input_preprocess
from sklearn.ensemble import GradientBoostingRegressor
from load_data import load_all
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor


def run(Input_data=[0,0,0,0,0,0,0,0,'Silica sand','bubbling fluidized bed'],Modle = 0):
    
    st.title("生物质流化床气化分析预测平台")

    # 获取绝对地址（不知道为什么用不了相对地址）
    current_path = inspect.getfile(inspect.currentframe())
    dir_name = os.path.dirname(current_path)
    file_abs_path = os.path.abspath(dir_name)[0:-4]
    print(file_abs_path)

    image = Image.open(file_abs_path+r'data/fluid_bed.gif')

    st.image(image, caption='流化床模拟')
    models = {"CO": AdaBoostRegressor(learning_rate=2, n_estimators=350),"H2":AdaBoostRegressor(learning_rate=2, n_estimators=350),"CH4": AdaBoostRegressor(learning_rate=2, n_estimators=400),"CO2":GradientBoostingRegressor(max_depth=1, min_samples_split=6, n_estimators=200),"GY":SVR(C=15, gamma=0.05)}
    if Modle == 0:
        st.subheader("请在侧边栏输入反应参数")
    elif Modle ==1:
        st.subheader("气体产出预测为：")
        input_data = Input_preprocess(Input_data)
        input_data_GY = Input_preprocess(Input_data,GY = 1)
        l = []
        l_T = []
        l_ER = []
        # 以后写个json文件装最优模型
        models = {"CO": AdaBoostRegressor(learning_rate=2, n_estimators=350),"H2":AdaBoostRegressor(learning_rate=2, n_estimators=350),"CH4": AdaBoostRegressor(learning_rate=2, n_estimators=400),"CO2":GradientBoostingRegressor(max_depth=1, min_samples_split=6, n_estimators=200),"GY":SVR(C=15, gamma=0.05)}
    
        
     
        values = []
        for target in ["CO", "H2", "CH4", "CO2","GY"]:
            X_train, X_test, y_train, y_test, train_data, test_data = load_all(target)
            model = models[target]
            #input_data = Input_preprocess(Input_data)
            if target == "GY":
                input_predict = model.fit(X_train, y_train).predict(input_data_GY)
            else:
                input_predict = model.fit(X_train, y_train).predict(input_data)
            l.append(input_predict)
        st.write("总产气量GY[Nm3/kg_daf]=",l[4][0])
        i = 0
        for target in ["CO", "H2", "CH4", "CO2"]:
            values.append(l[i][0] / sum(l)[0] * 100)
            #st.write(target, '[%vol_N2_free]=', l[i][0] / sum(l)[0] * 100, '%')
            i += 1
        
        pyplt=py.offline.plot
        labels=['CO[%vol_N2_free]','H2[%vol_N2_free]','CH4[%vol_N2_free]','CO2[%vol_N2_free]']
        
        trace=[go.Pie(labels=labels,values=values)]
        layout=go.Layout(
        title='产气比例图'
        )
        fig=go.Figure(data=trace,layout=layout)
        st.plotly_chart(fig, use_container_width=True)
        
        
        if st.checkbox('给出优化建议'):
            load_state = st.text('Loading...')
            dT = 200
            dER = 0.5
            for target in ["CO", "H2", "CH4", "CO2","GY"]:
                X_train, X_test, y_train, y_test, train_data, test_data = load_all(target)
                model = models[target]
                Input_data.columns=range(0,Input_data.shape[1])
                Input_data_T =  copy.deepcopy(Input_data)   # 浅拷贝深拷贝！！！！！！！！！
          
                Input_data_T[6] = Input_data[6]+dT    
                if target == "GY":
                    input_data_T = Input_preprocess(Input_data_T,GY = 1)
                else:
                    input_data_T = Input_preprocess(Input_data_T)
                input_predict_T = model.fit(X_train, y_train).predict(input_data_T)
                    
                l_T.append(input_predict_T)
                
                Input_data_ER =  copy.deepcopy(Input_data)
                Input_data_ER[5] = Input_data[5]+dER
                if target == "GY":
                    input_data_ER = Input_preprocess(Input_data_ER,GY = 1)
                else:
                    input_data_ER = Input_preprocess(Input_data_ER)
                input_predict_ER = model.fit(X_train, y_train).predict(input_data_ER)
                #input_data_ER = Input_preprocess(Input_data_ER)
                #input_predict_ER = model.fit(X_train, y_train).predict(input_data_ER)
                l_ER.append(input_predict_ER)
          

            i = 0
            for target in ["CO", "H2", "CH4", "CO2"]:
                if l_T[i]*l_T[4]>l[i]*l[4]:
                    str_T = '提高反应温度'
                else:
                    str_T = '降低反应温度'
                if l_ER[i]*l_ER[4]>l[i]*l[4]:
                    str_ER = ',提高当量比也许能增大产量哦'
                else:
                    str_ER = ',降低当量比也许能增大产量哦'
                str_ = str_T + str_ER
                st.write(target, '[Nm3/kg_daf]=', l[4][0]*l[i][0]/100,str_)
                i+=1


            load_state.text("loading...done")
    elif Modle == -1:
        st.subheader("气体产出预测为：")
        input_data = Input_preprocess(Input_data)
        input_data_GY = Input_preprocess(Input_data,GY = 1)
        l = []
        dic = {"CO":[],"H2":[],"CH4":[],"CO2":[]}
        # 以后写个json文件装最优模型
        
        load_state = st.text('Loading...')
        #st.write(input_data)
        for target in ["CO", "H2", "CH4", "CO2","GY"]:
            X_train, X_test, y_train, y_test, train_data, test_data = load_all(target)
            model = models[target]
            if target == "GY":
                input_predict = model.fit(X_train, y_train).predict(input_data_GY)
            else:
                input_predict = model.fit(X_train, y_train).predict(input_data)
            l.append(input_predict)
        l = pd.DataFrame(l).T
        #st.write(l)
        mean_r = l.mean(axis=1)
        #st.write(mean_r)
        #st.write(l[0][1])
        i = 0
        for target in ["CO", "H2", "CH4", "CO2"]:
            for j in range(0,len(l[0])):
                dic[target].append(l[i][j]*l[4][j]/100)
            i += 1
        #st.write(dic)
        df = pd.DataFrame(dic)
        #st.write(df)
        x = np.array(range(len(df)))
        y0 = np.array(dic["CO"])
        y1 = np.array(dic["H2"])
        y2 = np.array(dic["CH4"])
        y3 = np.array(dic["CO2"])
        # Create traces
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y0,
                            mode='lines+markers',
                            name='CO[%vol_N2_free]'))
        fig.add_trace(go.Scatter(x=x, y=y1,
                            mode='lines+markers',
                            name='H2[%vol_N2_free]'))
        fig.add_trace(go.Scatter(x=x, y=y2,
                            mode='lines+markers', name='CH4[%vol_N2_free]'))
        fig.add_trace(go.Scatter(x=x, y=y3,
                            mode='lines+markers', name='CO2[%vol_N2_free]'))
        st.plotly_chart(fig, use_container_width=True)
        st.write("预测结果下载：")
        st.download_button(
            label="Download data as CSV",
            data=df.to_csv(index = False).encode('utf-8'),
            file_name='PV.csv',
            mime='text/csv',
        )
        load_state.text("loading...done")
        

        st.subheader("试着在侧边栏选择模型调参吧！")

import copy
import streamlit as st
from PIL import Image
import os,inspect
from Input_preprocess import Input_preprocess
from sklearn.ensemble import GradientBoostingRegressor
from load_data import load_all
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor


def run(Input_data=[0,0,0,0,0,0,0,0,'Silica_sand','bubbling fluidized bed']):
    st.title("生物质流化床气化分析预测平台")

    # 获取绝对地址（不知道为什么用不了相对地址）
    current_path = inspect.getfile(inspect.currentframe())
    dir_name = os.path.dirname(current_path)
    file_abs_path = os.path.abspath(dir_name)[0:-4]
    print(file_abs_path)

    image = Image.open(file_abs_path+r'data/fluid_bed.gif')

    st.image(image, caption='流化床模拟')

    if Input_data == [0,0,0,0,0,0,0,0,'Silica_sand','bubbling fluidized bed']:
        st.subheader("请在侧边栏输入反应参数")
    else:

        st.subheader("气体产出预测为：")

        input_data = Input_preprocess(Input_data)
        l = []
        l_T = []
        l_ER = []
        # 以后写个json文件装最优模型
        models = {"CO": AdaBoostRegressor(learning_rate=2, n_estimators=350),"H2":AdaBoostRegressor(learning_rate=2, n_estimators=350),"CH4": AdaBoostRegressor(learning_rate=2, n_estimators=400),"CO2":GradientBoostingRegressor(max_depth=1, min_samples_split=6, n_estimators=200)}
        load_state = st.text('Loading...')
        if st.checkbox('给出优化建议'):
            dT = 200
            dER = 0.5
            for target in ["CO", "H2", "CH4", "CO2"]:
                X_train, X_test, y_train, y_test, train_data, test_data = load_all(target)
                model = models[target]
                
                input_data = Input_preprocess(Input_data)
                input_predict = model.fit(X_train, y_train).predict(input_data)
                l.append(input_predict)
                
                Input_data_T =  copy.deepcopy(Input_data)
                Input_data_T[6] = Input_data[6]+dT     
                input_data_T = Input_preprocess(Input_data_T)
                input_predict_T = model.fit(X_train, y_train).predict(input_data_T)
                l_T.append(input_predict_T)
                
                Input_data_ER =  copy.deepcopy(Input_data)
                Input_data_ER[5] = Input_data[5]+dER
                input_data_ER = Input_preprocess(Input_data_ER)
                input_predict_ER = model.fit(X_train, y_train).predict(input_data_ER)
                l_ER.append(input_predict_ER)
                print(Input_data)



            #  加起来求百分比 （一看加起来就不得百分百就尴尬了呀）
            i = 0
            for target in ["CO", "H2", "CH4", "CO2"]:
                if l_T[i]>l[i]:
                    str_T = '提高反应温度'
                else:
                    str_T = '降低反应温度'
                if l_ER[i]>l[i]:
                    str_ER = ',提高当量比也许能增大占比哦'
                else:
                    str_ER = ',降低当量比也许能增大占比哦'
                str_ = str_T + str_ER
                st.write(target, '[%vol_N2_free]=', l[i][0]/sum(l)[0]*100,'%—————',str_)
                i+=1

        else:
            for target in ["CO", "H2", "CH4", "CO2"]:
                X_train, X_test, y_train, y_test, train_data, test_data = load_all(target)
                model = models[target]
                input_data = Input_preprocess(Input_data)
                input_predict = model.fit(X_train, y_train).predict(input_data)
                l.append(input_predict)
            i = 0
            for target in ["CO", "H2", "CH4", "CO2"]:
                st.write(target, '[%vol_N2_free]=', l[i][0] / sum(l)[0] * 100, '%')
                i += 1
        load_state.text("loading...done")

        st.subheader("试着在侧边栏选择模型调参吧！")

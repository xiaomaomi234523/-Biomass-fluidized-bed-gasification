from sklearn.ensemble import AdaBoostRegressor
import streamlit as st
from load_data import load_all
from plot_learning_curve import Plot_learning_curve
import numpy as np

from Input_preprocess import Input_preprocess


def run(Input_data=[0,0,0,0,0,0,0,0,'Silica_sand','bubbling fluidized bed'],Modle = 0):
    st.title('Adaboost')

    # 控件
    target = st.sidebar.selectbox("选择要预测的气体", (
        "CO", "H2", "CH4", "CO2"))


    n_estimators = st.slider('n_estimators', 1, 1000, 200)
    learning_rate = st.slider("learning_rate", 1, 10, 1)

    # 加载数据
    X_train, X_test, y_train, y_test, train_data, test_data = load_all(target)


    # 学习曲线

    model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate)

    if st.checkbox('学习曲线'):
          st.write('集成学习绘制学习曲线占用资源过多暂时停用,请试试单一学习器')
#         curve_load_state = st.text('Loading learning curve...')
#         title = r"Learning Curves GBDT"
#         Plot_learning_curve(model, title, X_train, y_train)
#         curve_load_state.text('Loading learning curve...done!')

    # 评价指标
    test_predict = model.fit(X_train, y_train).predict(X_test).reshape(-1, 1)
    print(test_predict)
    rmse = np.mean((test_predict - y_test) ** 2) ** 0.5  ## 均方根误差
    test_mape = np.mean(abs((y_test - test_predict) / y_test) * 100 / y_test.shape[1])  # 平均绝对百分误差

    st.title("在测试集上的表现")
    st.write("均方根误差：",rmse)
    st.write("平均绝对百分误差",test_mape)

    #  根据输入数据预测
    import pandas as pd
    if Modle == 0:
        st.subheader("请在侧边栏输入反应参数")
    elif Modle == 1:
        input_data = Input_preprocess(Input_data)
        input_predict = model.fit(X_train, y_train).predict(input_data)
        st.subheader('气体产出预测为：')
        st.write(target, '[%vol_N2_free]=', input_predict[0])
    elif Modle ==-1:
        st.subheader("气体产出预测为：")
        input_data = Input_preprocess(Input_data)
        l = []
        load_state = st.text('Loading...')
        #st.write(input_data)
        input_data = Input_preprocess(Input_data)
        input_predict = model.fit(X_train, y_train).predict(input_data)
        l.append(input_predict)
        l = pd.DataFrame(l).T
        dic = {target+"[%vol_N2_free]":l[0]}
        df = pd.DataFrame(dic)
        st.write(df)
    load_state.text("loading...done")

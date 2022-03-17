import streamlit as st
from load_data import load_all
from sklearn.ensemble import GradientBoostingRegressor
from plot_learning_curve import Plot_learning_curve
import numpy as np
from Input_preprocess import Input_preprocess

def run(Input_data=[0,0,0,0,0,0,0,0,'Silica_sand','bubbling fluidized bed']):
    st.title('GBDT')

    # 控件
    target = st.sidebar.selectbox("选择要预测的气体", (
        "CO", "H2", "CH4", "CO2"))


    n_estimators = st.slider('n_estimators', 1, 1000, 200)
    max_depth = st.slider("max_depth", 1, 10, 1)
    min_samples_split = st.slider('min_samples_split', 1, 10, 6)

    # 加载数据
    X_train, X_test, y_train, y_test, train_data, test_data = load_all(target)


    # 学习曲线
    model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_split=min_samples_split)

    if st.checkbox('学习曲线'):
        st.write('集成学习绘制学习曲线占用资源过多暂时停用,请试试单一学习器')
        #curve_load_state = st.text('Loading learning curve...')
        #title = r"Learning Curves GBDT"
        #Plot_learning_curve(model, title, X_train, y_train)
        #curve_load_state.text('Loading learning curve...done!')

    # 评价指标
    test_predict = model.fit(X_train, y_train).predict(X_test).reshape(-1, 1)
    print(test_predict,"加载成功")
    rmse = np.mean((test_predict - y_test) ** 2) ** 0.5  ## 均方根误差
    test_mape = np.mean(abs((y_test - test_predict) / y_test) * 100 / y_test.shape[1])  # 平均绝对百分误差

    st.title("在测试集上的表现")
    st.write("均方根误差：",rmse)
    st.write("平均绝对百分误差",test_mape)

    #  根据输入数据预测

    if Input_data != [0, 0, 0, 0, 0, 0, 0, 0, 'Silica_sand', 'bubbling fluidized bed']:
        #  根据输入数据预测
        input_data = Input_preprocess(Input_data)
        input_predict = model.fit(X_train, y_train).predict(input_data)
        st.subheader('**预测值**')
        st.write(target, '[%vol_N2_free]=', input_predict[0])
    else:
        st.subheader('**请在侧边栏上传文件或输入数据**')

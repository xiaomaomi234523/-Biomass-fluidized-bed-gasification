import streamlit as st
from load_data import load_all
from sklearn.svm import SVR
from plot_learning_curve import Plot_learning_curve
import numpy as np
from Input_preprocess import Input_preprocess


def run(Input_data=[0,0,0,0,0,0,0,0,'Silica_sand','bubbling fluidized bed']):
    st.title('SVR')

    # 控件
    target = st.sidebar.selectbox("选择要预测的气体", (
        "CO", "H2", "CH4", "CO2"))

    kernel = st.selectbox("选择核", (
        "rbf","sigmoid" ,"poly" ))
    if kernel == "rbf":
        C = st.slider('C', 1, 100, 60)
        gamma = 0.01*st.slider(chr(947)+"*10^2", 1, 500, 25)
    elif kernel == "poly":
        C = st.slider('C', 1, 2000, 1000)
        gamma = 0.01 * st.slider(chr(947) + "*10^2", 1, 500, 25)
        st.write("多项式核函数可以实现将低维的输入空间映射到高纬的特征空间，但是多项式核函数的参数多，当多项式的阶数比较高的时候，核矩阵的元素值将趋于无穷大或者无穷小，计算复杂度会大到无法计算。")
        st.subheader("绘制学习曲线占用资源太多，试试其它模型吧")
    elif kernel == "sigmoid":
        C = st.slider('C', 1, 10000, 2000)
        gamma = 0.00001 * st.slider(chr(947)+"*10^5" , 1, 100, 50)


    # 加载数据
    X_train, X_test, y_train, y_test, train_data, test_data = load_all(target)


    # 学习曲线

    model = SVR(kernel=kernel, C=C, gamma=gamma)
    if kernel != 'poly' :
        title = r"Learning Curves (SVM, " + kernel + " kernel, "+ chr(947) + "=" + str(gamma) + ")"
        Plot_learning_curve(model,title,X_train,y_train)

    # 评价指标
    test_predict = model.fit(X_train, y_train).predict(X_test).reshape(-1, 1)
    rmse = np.mean((test_predict - y_test) ** 2) ** 0.5  ## 均方根误差
    test_mape = np.mean(abs((y_test - test_predict) / y_test) * 100 / y_test.shape[1])  # 平均绝对百分误差

    st.title("在测试集上的表现")
    st.write("均方根误差：", rmse)
    st.write("平均绝对百分误差", test_mape)

    #  根据输入数据预测

    if Input_data != [0, 0, 0, 0, 0, 0, 0, 0, 'Silica_sand', 'bubbling fluidized bed']:
        #  根据输入数据预测
        input_data = Input_preprocess(Input_data)
        input_predict = model.fit(X_train, y_train).predict(input_data)
        st.subheader('**预测值**')
        st.write(target, '[%vol_N2_free]=', input_predict[0])
    else:
        st.subheader('**请在侧边栏上传文件或输入数据**')

from sklearn.metrics import mean_squared_error, mean_absolute_error,median_absolute_error
from keras.utils.np_utils import to_categorical#import np_utils
from keras.utils import np_utils
#import matplotlib.pyplot as plt
from dateutil.parser import parse
from datetime import datetime
import pandas as pd
import numpy as np
from math import sqrt
from tkinter import _flatten
from keras.models import Model,load_model
from keras.layers import CuDNNLSTM, Dense,Input,concatenate,Reshape
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
#from keras.utils.vis_utils import plot_model
import time, math
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.backend.tensorflow_backend import set_session
os.environ['CUDA_VISIBLE_DEVICES']='5'
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   
sess = tf.Session(config=config)
KTF.set_session(sess)
T=1
a=os.listdir('all/')
a.sort()
a=a[51:60]
result=pd.DataFrame([])
flow=pd.DataFrame([])
def valuate(target,prediction,a):
    #
    rmse=np.sqrt(mean_squared_error(target,prediction))
    
    mae=median_absolute_error(target,prediction)
    
    mae_old=mean_absolute_error(target,prediction)
    
    mape=np.mean(np.abs((prediction - target) / target)) * 100
       
    print('---------------------')
    print(rmse)#均方根误差RMSE
    a.append(rmse)
    
    print(mae)#median_absolute_error
    a.append(mae)
    
    print(mae_old)#平均绝对误差 
    a.append(mae_old)
    
    print(mape)#平均绝对误差 
    a.append(mape)    
    
    print('---------------------')
    a.append(0)
    
def data_re(data):
    start=datetime(2013,1,1,0,0,0)
    data_t=data['a']
    data_flow=data['j']
    data_t=[parse(x) for x in data_t]
    delta=data_t[1]-data_t[0]
    data_rep=[]
    for i in range(len(data)):
        if data_t[i]==start:
            data_rep.append([start,data_flow[i]])
            start+=delta
        else:
            T=start
            x=data_t[i]
            c=(x-T)/delta
            c=int(c)
            for j in range(int((data_t[i]-T)/delta)):
                
                if i< 1 :
                    
                    data_rep.append([start,data_flow[i]])
                    start+=delta
                else:
                    
                    data_rep.append([start,(data_flow[i-1]+data_flow[i])/2])
                    start+=delta
            data_rep.append([start,data_flow[i]])
            start+=delta      
            
    c=[]        
    [c.append(x) for x in data_rep]
    c=pd.DataFrame(c,columns=['date','flow']) 
    num=list(range(0,288))*243
    c['num']=num
    c=c[c['num'].isin(list(range(73,265)))]     
    return c
def charu_xingqi(a):
 
    N=2
    xq=[]

    for i in range(243):
        for j in range(len(a)//243):
            xq.append(N)
        N+=1
        if N>7:
            N=1
    a['xingqi']=xq

    return a


def flow_reshape(data,n,j=1,axi=1):# traffic flow aggregation,5min,10min ,or 15min
    data=data.reshape(-1,n)
    data=data.sum(axis=axi)/j
    return data

def train_data(data,samples,day,timesteps=9):
    N=0
    cdata=[]
    for i in range(day):
        for j in range(samples-timesteps):
            cdata.append(data[N:N+timesteps+1])
            N+=1
        N+=(timesteps)
    return cdata



def show(predictions,mm,y_val1,y_val2,y_val3,y_val4,a):# plot 
    
    k=[]
    prediction1=predictions[0]
    prediction2=predictions[1]
    prediction3=predictions[2]
    prediction4=predictions[3]
    
    prediction1=mm.inverse_transform(prediction1)
    prediction2=mm.inverse_transform(prediction2)
    prediction3=mm.inverse_transform(prediction3)
    prediction4=mm.inverse_transform(prediction4)
    
    test_y1=mm.inverse_transform(y_val1)  
    test_y2=mm.inverse_transform(y_val2)
    test_y3=mm.inverse_transform(y_val3)  
    test_y4=mm.inverse_transform(y_val4)  

    pp1=prediction1[:,0]
    pp2=prediction2[:,0]
    pp3=prediction3[:,0]
    pp4=prediction4[:,0]
    
    flow[a]=list(pp1)+list(pp2)+list(pp3)+list(pp4)
    
    valuate(test_y1,prediction1,k) 
    valuate(test_y2,prediction2,k) 
    valuate(test_y3,prediction3,k) 
    valuate(test_y4,prediction4,k) 
    result[a]=k
    #result.to_csv('mymodel_T1_3.CSV')
    flow.to_csv('flow_mymodel_T1_33.CSV')
       
#    plt.figure()    
#    x=range(len(test_y1))
#    plt.plot(x,test_y1, 'g+',label='ground truth')
#    plt.plot(x,prediction1, 'r+',label='test value')
#    plt.legend()
#    plt.show()
#    plt.figure()    
#    x=range(len(test_y2))
#    plt.plot(x,test_y2, 'g+',label='ground truth')
#    plt.plot(x,prediction2, 'r+',label='test value')
#    plt.legend()
#    plt.show()
#    
#    plt.figure()    
#    x=range(len(test_y3))
#    plt.plot(x,test_y3, 'g+',label='ground truth')
#    plt.plot(x,prediction3, 'r+',label='test value')
#    plt.legend()
#    plt.show()
#    
#    plt.figure()    
#    x=range(len(test_y4))
#    plt.plot(x,test_y4, 'g+',label='ground truth')
#    plt.plot(x,prediction4, 'r+',label='test value')
#    plt.legend()
#    plt.show()
    
def model1(dim_flow,dim_label):
                        # expected input data shape: (batch_size, timesteps, data_dim)
    Input_shape=(dim_flow,1)
    Input_shape1=(dim_label,1)
    Input_shape2=(dim_label,1)
    Input_shape3=(dim_label,1)
    Input_shape4=(dim_label,1)
    
    X=Input(Input_shape)
    X1=Input(Input_shape1)
    X2=Input(Input_shape2)
    X3=Input(Input_shape3)
    X4=Input(Input_shape4)

    
    X5=CuDNNLSTM(60,return_sequences=True)(X)
    X5=CuDNNLSTM(30,return_sequences=True)(X5)
    X5=CuDNNLSTM(15,)(X5) 
    
    share1=CuDNNLSTM(30,return_sequences=True)
    share2=CuDNNLSTM(5)
    X6=share1(X1)     
    X6=share2(X6)
    
    X7=share1(X2)     
    X7=share2(X7)
    
    X8=share1(X3)     
    X8=share2(X8)
    
    X9=share1(X4)     
    X9=share2(X9)    
    
    X13=concatenate([X5, X6])
    X14=concatenate([X5, X7])
    X15=concatenate([X5, X8])
    X16=concatenate([X5, X9])

    X17=Dense(1,activation='sigmoid')(X13)
    X18=Dense(1,activation='sigmoid')(X14)
    X19=Dense(1,activation='sigmoid')(X15)
    X20=Dense(1,activation='sigmoid')(X16)
    model = Model(inputs =[X,X1,X2,X3,X4],outputs = [X17,X18,X19,X20], name = 'my model')
   

    model.compile(loss='mean_squared_error',
                  optimizer='adam')
#                  ,metrics=['accuracy'])
#    plot_model(model,to_file='my model9.png',show_shapes=True)
#    model.summary()
    return model    
    
def model3(dim_flow,dim_label):
                        # expected input data shape: (batch_size, timesteps, data_dim)
    Input_shape=(dim_flow,1)
    Input_shape1=(dim_label,1)
    Input_shape2=(dim_label,1)
    Input_shape3=(dim_label,1)
    Input_shape4=(dim_label,1)
    
    X=Input(Input_shape)
    X1=Input(Input_shape1)
    X2=Input(Input_shape2)
    X3=Input(Input_shape3)
    X4=Input(Input_shape4)
    
    X5=CuDNNLSTM(60,return_sequences=True)(X)
    X5=CuDNNLSTM(30,return_sequences=True)(X5)
    X5=CuDNNLSTM(15,)(X5) 
    

    X6=CuDNNLSTM(30,return_sequences=True)(X1)     
    X6=CuDNNLSTM(5)(X6)
    
    X7=CuDNNLSTM(30,return_sequences=True)(X2)     
    X7=CuDNNLSTM(5)(X7)
    
    X8=CuDNNLSTM(30,return_sequences=True)(X3)     
    X8=CuDNNLSTM(5)(X8)
    
    X9=CuDNNLSTM(30,return_sequences=True)(X4)     
    X9=CuDNNLSTM(5)(X9)
    
    X13=concatenate([X5, X6])
    X17=Dense(1,activation='sigmoid')(X13)
    
    X14=concatenate([X5,X17])
    X14=Reshape([-1,1])(X14)
    X14=CuDNNLSTM(15)(X14)
    X14=concatenate([X14,X7])
    X18=Dense(1,activation='sigmoid')(X14)
    
    X15=concatenate([X5,X17,X18])
    X15=Reshape([-1,1])(X15)
    X15=CuDNNLSTM(15)(X15)
    X15=concatenate([X15,X8])
    X19=Dense(1,activation='sigmoid')(X15)
    
    X16=concatenate([X5,X17,X18,X19])
    X16=Reshape([-1,1])(X16)
    X16=CuDNNLSTM(15)(X16)
    X16=concatenate([X16,X9])    
    X20=Dense(1,activation='sigmoid')(X16)
        
    model = Model(inputs =[X,X1,X2,X3,X4],outputs = [X17,X18,X19,X20], name = 'my model')
   

    model.compile(loss='mean_squared_error',
                  optimizer='adam')
#                  ,metrics=['accuracy'])
#    plot_model(model,to_file='my model9.png',show_shapes=True)
#    model.summary()
    return model



def main(k):
    data1=pd.read_csv('all/'+k)
    a1=data_re(data1)#data repair
    a1=charu_xingqi(a1)#set week label
    num_day=len(a1)//243

    #删除周六日
    a_week1=a1[~a1['xingqi'].isin([6,7])]#delete weekend data
    
    a_week1['date'] = pd.to_datetime(a_week1.date)
    rng = pd.date_range(pd.to_datetime('2013-1-1'),pd.to_datetime('2013-1-2'), freq='5min')
    a_week2=a_week1[~a_week1['date'].isin(rng)]
    
    rng = pd.date_range(pd.to_datetime('2013-1-23'),pd.to_datetime('2013-1-24'), freq='5min')
    a_week2=a_week2[~a_week2['date'].isin(rng)]
    
    rng = pd.date_range(pd.to_datetime('2013-2-18'),pd.to_datetime('2013-1-19'), freq='5min')
    a_week2=a_week2[~a_week2['date'].isin(rng)]    
    
    days=len(a_week2)//num_day    
    
     #T=3 15min;T=2 10min; T=1 5min
    data1=flow_reshape(a_week2['flow'].values,T,j=1)# traffic flow aggregation,5min,10min ,or 15min
    
    mm = MinMaxScaler()
    data1=mm.fit_transform(data1.reshape(-1,1))   
    data1=data1.tolist()   
    num_oneday=len(data1)//days  #number of samples per day
    
    data_train=train_data(data=data1,samples=num_oneday,day=days) #make dataset   
    
    data_train=np.array(data_train).reshape(-1,10,1)
    #make time label
    num=len(data_train)//days
    I=list(range(num_oneday))*days
    
    I_train=np.array(train_data(data=I,samples=num_oneday,day=days)).reshape(-1,10,1)
    

    dim_flow=6
    I1=I_train[:,6]-dim_flow
    I2=I_train[:,7]-dim_flow
    I3=I_train[:,8]-dim_flow
    I4=I_train[:,9]-dim_flow
    
    dim_label=num_oneday-dim_flow 
    
    I1=np_utils.to_categorical(I1,dim_label).reshape(-1,dim_label,1)
    I2=np_utils.to_categorical(I2,dim_label).reshape(-1,dim_label,1)
    I3=np_utils.to_categorical(I3,dim_label).reshape(-1,dim_label,1)
    I4=np_utils.to_categorical(I4,dim_label).reshape(-1,dim_label,1)
     
    #split training set and test set
    
    x_train = data_train[:num*112,:6]
    
    x_train1= I1[:num*112,:]
    x_train2= I2[:num*112,:]
    x_train3= I3[:num*112,:]
    x_train4= I4[:num*112,:]
    
    y_train1 = data_train[:num*112,6]
    y_train2 = data_train[:num*112,7]
    y_train3 = data_train[:num*112,8]
    y_train4 = data_train[:num*112,9]
    
    
    #val_dataset
    x_val = data_train[num*112:num*142,:6]
    
    x_val1 = I1[num*112:num*142,:]
    x_val2 = I2[num*112:num*142,:]
    x_val3 = I3[num*112:num*142,:]
    x_val4 = I4[num*112:num*142,:]
    
    y_val1 = data_train[num*112:num*142,6]
    y_val2 = data_train[num*112:num*142,7]
    y_val3= data_train[num*112:num*142,8]
    y_val4= data_train[num*112:num*142,9]
    
    #test_dataset
    
    x_test = data_train[num*142:num*172,:6]
    
    x_test1 = I1[num*142:num*172,:]
    x_test2 = I2[num*142:num*172,:]
    x_test3 = I3[num*142:num*172,:]
    x_test4 = I4[num*142:num*172,:]
    
    y_test1 = data_train[num*142:num*172,6]
    y_test2 = data_train[num*142:num*172,7]
    y_test3= data_train[num*142:num*172,8]
    y_test4= data_train[num*142:num*172,9]
    
    checkpointer = ModelCheckpoint(filepath='./weights/mymodel_T1_3.h5',monitor='val_loss', verbose=1,save_weights_only=True, save_best_only=True)   
    model=model3(dim_flow,dim_label)
    since1=time.time()
    history=model.fit([x_train,x_train1,x_train2,x_train3,x_train4],[y_train1,y_train2,y_train3,y_train4],
              batch_size=64, epochs=100, shuffle=True,
              validation_data=([x_val,x_val1,x_val2,x_val3,x_val4],[y_val1,y_val2,y_val3,y_val4]),callbacks=[checkpointer]) 
    
    now1=time.time()
    s1=now1-since1   
    print('train： %s' % s1)  

    model.load_weights("./weights/mymodel_T1_3.h5") 
    
    since2=time.time()    
    predictions=model.predict([x_test,x_test1,x_test2,x_test3,x_test4])   
    now2=time.time()
    s2=now2-since2
    print('test: %s' % s2)     
       
#    plt.plot(history.history['loss'], label='train loss')
#    plt.plot(history.history['val_loss'], label='test loss')
#    plt.legend()
#    plt.show()
#    print('-----------------------------------------')
    show(predictions,mm,y_test1,y_test2,y_test3,y_test4,k)
#    
    
for i in a:
    print('station:'+i)
    print("+++++++++++++++++++++++")
    main(i)
    print("+++++++++++++++++++++++")
#result.to_csv('./result/mymodel.CSV')

#result=pd.DataFrame([])
#for i in a:
#    print('station:'+i)
#    print("+++++++++++++++++++++++")
#    main(i)
#    print("+++++++++++++++++++++++")
#result.to_csv('duobu-model3-result-2.CSV.CSV')
#    
#    

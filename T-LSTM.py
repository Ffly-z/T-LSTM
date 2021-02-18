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
os.environ['CUDA_VISIBLE_DEVICES']='6'
a=os.listdir('dataset/')
a.sort()
result=pd.DataFrame([])
#flow=pd.DataFrame([])
def valuate(target,prediction,a):#calculate the average error
    #
    
    target=target.tolist()
    prediction=prediction.tolist()
    target=list(_flatten(target))
    prediction=list(_flatten(prediction))
    error = []
    for i in range(len(target)):
        #
        error.append(target[i] - prediction[i]) 

    squaredError = []
    absError = []
    absPError=[]
    for i in range(len(error)):
        #
        squaredError.append(error[i] * error[i])#target-prediction之差平方 
        absError.append(abs(error[i]))#误差绝对值
        absPError.append(abs(error[i])/target[i])
    print('---------------------')
    print(sqrt(sum(squaredError) / len(squaredError)))#均方根误差RMSE
    a.append(sqrt(sum(squaredError) / len(squaredError)))
    print(sum(absError) / len(absError))#平均绝对误差MAE
    a.append(sum(absError) / len(absError))
    print(sum(absPError) / len(absError))#平均相对误差 
    a.append(sum(absPError) / len(absError))
    print('---------------------')
    a.append(0)
    
def data_re(data):#data repair
    start=datetime(2016,5,1,0,5,0)
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
                data_rep.append([start,(data_flow[i-1]+data_flow[i])/2])
                start+=delta
            data_rep.append([start,data_flow[i]])
            start+=delta      
            
    if len(data)<8928:
        c=[[1,2]]
    else:
        c=[]
    [c.append(x) for x in data_rep]
    c=pd.DataFrame(c,columns=['date','flow']) 
    num=list(range(0,288))*31
    c['num']=num
    c=c[c['num'].isin(list(range(73,265)))]     
    return c
def charu_xingqi(a):#set week label
    
    N=7
    xq=[]
    b=1
    c=[]
    for i in range(31):
        for j in range(len(a)//31):
            xq.append(N)
            c.append(b)
        b=b+1
        N+=1
        if N>7:
            N=1
    a['xingqi']=xq
    a['riqi']=c
    return a


def flow_reshape(data,n,j=1,axi=1):# traffic flow aggregation,5min,10min ,or 15min
    data=data.reshape(-1,n)
    data=data.sum(axis=axi)/j
    return data

def train_data(data,samples,timesteps=9,days=21): # make data set
    N=0
    cdata=[]
    for i in range(days):
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

    #pp=prediction1[:,0]
    #flow[a]=list(pp)
    
    valuate(test_y1,prediction1,k) 
    valuate(test_y2,prediction2,k) 
    valuate(test_y3,prediction3,k) 
    valuate(test_y4,prediction4,k) 
    result[a]=k
    result.to_csv('mymodel_mae.CSV')
    #flow.to_csv('flow_model2.CSV')
       
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
    
def model2(dim_flow,dim_label):
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
   

    model.compile(loss='mean_absolute_error',
                  optimizer='adam')
#                  ,metrics=['accuracy'])
#    plot_model(model,to_file='my model9.png',show_shapes=True)
#    model.summary()
    return model

def model3( ):
                        # expected input data shape: (batch_size, timesteps, data_dim)
    Input_shape=(6,1)
    Input_shape1=(58,1)
    Input_shape2=(58,1)
    Input_shape3=(58,1)
    Input_shape4=(58,1)
    
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
    X17=Dense(1,activation='sigmoid')(X13)
    
    X14=concatenate([X5,X17])
    X14=Reshape([-1,1])(X14)
    X14=CuDNNLSTM(30,return_sequences=True)(X14)
    X14=CuDNNLSTM(15)(X14)
    X14=concatenate([X14,X7])
    X18=Dense(1,activation='sigmoid')(X14)
    
    X15=concatenate([X5,X17,X18])
    X15=Reshape([-1,1])(X15)
    X15=CuDNNLSTM(30,return_sequences=True)(X15)
    X15=CuDNNLSTM(15)(X15)
    X15=concatenate([X15,X8])
    X19=Dense(1,activation='sigmoid')(X15)
    
    X16=concatenate([X5,X17,X18,X19])
    X16=Reshape([-1,1])(X16)
    X16=CuDNNLSTM(30,return_sequences=True)(X16)   
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
    #删除周六日
    a_week1=a1[~a1['xingqi'].isin([6,7])]#delete weekend data
    a_week2=a_week1[~a_week1['riqi'].isin([30])]#delete holiday data
    
    T=3 #T=3 15min;T=2 10min; T=1 5min
    data1=flow_reshape(a_week2['flow'].values,T,j=1)# traffic flow aggregation,5min,10min ,or 15min
    
    mm = MinMaxScaler()
    data1=mm.fit_transform(data1.reshape(-1,1))   
    data1=data1.tolist()   
    num_oneday=len(data1)//21#number of samples per day
    
    data_train=train_data(data=data1,samples=num_oneday)#make dataset   
    data_train=np.array(data_train).reshape(-1,10,1)
    #make time label
    num=len(data_train)//21
    I=list(range(num_oneday))*21
    I_train=np.array(train_data(data=I,samples=num_oneday)).reshape(-1,10,1)
  
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
    x_train = data_train[:num*17,:dim_flow] 
    x_train1= I1[:num*17,:]
    x_train2= I2[:num*17,:]
    x_train3= I3[:num*17,:]
    x_train4= I4[:num*17,:]
    
    y_train1 = data_train[:num*17,6]
    y_train2 = data_train[:num*17,7]
    y_train3 = data_train[:num*17,8]
    y_train4 = data_train[:num*17,9]
    
    x_test = data_train[num*17:,:6]
    
    x_test1 = I1[num*17:,:]
    x_test2 = I2[num*17:,:]
    x_test3 = I3[num*17:,:]
    x_test4 = I4[num*17:,:]
    
    y_test1 = data_train[num*17:,6]
    y_test2 = data_train[num*17:,7]
    y_test3= data_train[num*17:,8]
    y_test4= data_train[num*17:,9]
    
    checkpointer = ModelCheckpoint(filepath='./wights/mymodelT_2.h5',monitor='val_loss', verbose=1,save_weights_only=True, save_best_only=True)   
    model=model1(dim_flow,dim_label)

    history=model.fit([x_train,x_train1,x_train2,x_train3,x_train4],[y_train1,y_train2,y_train3,y_train4],
              validation_split=0.1,batch_size=num, epochs=500, shuffle=True,
              callbacks=[checkpointer])  
  
    model.load_weights("./wights/mymodelT_2.h5") 
    predictions=model.predict([x_test,x_test1,x_test2,x_test3,x_test4])    
  
    
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    


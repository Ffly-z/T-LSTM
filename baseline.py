#baseline
def LSTM(input_dim=1 ,out_dim=4,timesteps =6,unit1=60,unit2=30,unit3=15):
                        # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(Bidirectional(CuDNNLSTM(unit1, return_sequences=True),input_shape=(timesteps, input_dim)))  # returns a sequence of vectors of dimension 32
#    model.add(Dropout(0.3))
    model.add(Bidirectional(CuDNNLSTM(unit2, return_sequences=True)))  # returns a sequence of vectors of dimension 32
#    model.add(Dropout(0.3))
    model.add(Bidirectional(CuDNNLSTM(unit3)))                # return a single vector of dimension 32
#    model.add(Dropout(0.3))
    model.add(Dense(out_dim, activation='sigmoid'))

    model.compile(loss='mean_squared_error',
                  optimizer='adam')
#                  ,metrics=['accuracy'])
    
    return model

def GRU(input_dim=1 ,out_dim=4,timesteps =6,unit1=60,unit2=30,unit3=20):
                        # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(CuDNNGRU(unit1, return_sequences=True,
                   input_shape=(timesteps, input_dim)))  # returns a sequence of vectors of dimension 32
#    model.add(Dropout(0.3))
    model.add(CuDNNGRU(unit2, return_sequences=True))  # returns a sequence of vectors of dimension 32
#    model.add(Dropout(0.3))
    model.add(CuDNNGRU(unit3))                # return a single vector of dimension 32
#    model.add(Dropout(0.3))
    model.add(Dense(out_dim, activation='sigmoid'))

    model.compile(loss='mean_squared_error',
                  optimizer='adam')
#                  ,metrics=['accuracy'])
    
    return model
def BiLSTM(input_dim=1 ,out_dim=4,timesteps =6,unit1=60,unit2=40,unit3=20):
                        # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(CuDNNLSTM(unit1, return_sequences=True,
                   input_shape=(timesteps, input_dim)))  # returns a sequence of vectors of dimension 32
#    model.add(Dropout(0.3))
    model.add(CuDNNLSTM(unit2, return_sequences=True))  # returns a sequence of vectors of dimension 32
#    model.add(Dropout(0.3))
    model.add(CuDNNLSTM(unit3))                # return a single vector of dimension 32
#    model.add(Dropout(0.3))
    model.add(Dense(out_dim, activation='sigmoid'))

    model.compile(loss='mean_squared_error',
                  optimizer='adam')
#                  ,metrics=['accuracy'])
    
    return model
 def CNN( ):
    input_shape = (6,1)
    X= Input(input_shape)  
    X1=Conv1D(64,2,strides=3, padding='same',data_format='channels_last',activation='relu')(X)
    X1=Conv1D(filters=32, kernel_size=1,strides=2, data_format='channels_last',padding='same',activation='relu')(X1)    
    X1=Conv1D(filters=16, kernel_size=1,strides=2, data_format='channels_last',padding='same',activation='relu')(X1)    
    X1=MaxPooling1D(2,padding='same', data_format='channels_last')(X1)
    X1=Flatten()(X1)
    X1=Dense(4,activation='sigmoid')(X1)
    model = Model(inputs = X, outputs = X1, name = 'CNN')
    model.compile(loss='mean_squared_error',
                  optimizer='sgd')
    
    def SAE( ):#SAE
    input_shape = (6,)
    X = Input(input_shape)  
    X1=Dense(300,activation='sigmoid')(X)
    X1=Dense(100,activation='sigmoid')(X1)
    X1=Dense(300,activation='sigmoid')(X1)
    X1=Dense(4,activation='sigmoid')(X1)
    model = Model(inputs = X, outputs = X1, name = 'CNN-LSTM')
    model.compile(loss='mean_squared_error',
                  optimizer='adam')
    return model
  
 svr_rbf1 = SVR(kernel='rbf', C=100, gamma=0.8 )
  
 def random_walk(data):
 #time span
    T=2
    #drift factor飘移率
    mu=0.1 
    #volatility波动率
    sigma=0.04 
    #t=0初试价
    #length of steps
    dt=0.01 
    N=100
    t=np.linspace(0,T,N)
    
    #布朗运动
    W=np.random.standard_normal(size=N)
    W=np.cumsum(W)*np.sqrt(dt)
    
    X=(mu-0.5*sigma**2)*t+sigma*W
    S=data*np.exp(X)
    
    return S[:,:4]
    
    
    return model 
  def model1( ):
    input_shape = (6,1)
    X = Input(input_shape)  
    X1=Conv1D(60,2,padding='same',data_format='channels_last',activation='relu')(X)
    X1=Conv1D(30,2,data_format='channels_last',padding='same',activation='relu')(X1)    
    X1=CuDNNLSTM(30,return_sequences=True)(X1)
    X1=CuDNNLSTM(15)(X1)
    X1=Dense(4,activation='sigmoid')(X1)
    model = Model(inputs = X, outputs = X1, name = 'CNN-LSTM')
    model.compile(loss='mean_squared_error',
                  optimizer='adam')
    return model    
  
  

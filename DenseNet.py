import numpy as np 
import pandas as pd
import math
from sklearn import metrics
from matplotlib import pyplot as plt

def Params(Func,Y,MP = True):
    params = {}
    params['proc']=3
    if MP == False:
        params['proc']=1
    if Func == 'Full':
        epochs = 200
        K = 20
        splits_per_mod = 5
        N = np.arange(2,15,1,dtype='int32')**1.5
    elif Func == 'Test':
        epochs = 100
        K = 4
        splits_per_mod = 2
        N = np.arange(2,8,2,dtype='int32')**2
    N = np.repeat(N,K)
    d = {'N':N.astype(int)}
    Runs = pd.DataFrame(data=d)
    Runs['MSE'] = 0.0
    Runs['R2'] = 0.0
    Runs['Model']=0
    params['K'] = K
    params['epochs'] = epochs
    params['Y'] = Y
    params['splits_per_mod'] = splits_per_mod
    params['Save'] = {}
    params['Save']['Weights']=False
    params['Save']['Model']=False
    
    return(Runs,params)

def Dense_Model(params,inputs,lr=1e-4,Memory=.9):
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasRegressor
    from keras.callbacks import EarlyStopping,ModelCheckpoint,LearningRateScheduler
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = Memory
    session = tf.Session(config=config)
    initializer = keras.initializers.glorot_uniform(seed=params['iteration'])
    model = Sequential()
    model.add(Dense(params['N'], input_dim=inputs,activation='relu',kernel_initializer=initializer))
    model.add(Dense(1))
    NUM_GPU = 1 # or the number of GPUs available on your machine
    adam = keras.optimizers.Adam(lr = lr)
    gpu_list = []
    for i in range(NUM_GPU): gpu_list.append('gpu(%d)' % i)
    model.compile(loss='mean_squared_error', optimizer='adam')#,context=gpu_list) # - Add if using MXNET
    if params['Save']['Weights'] == True:
        callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint(filepath=params['Y']+'/Weights/'+str(params['Model'])+'_'+str(params['iteration'])+'_'+str(params['seed'])+'.h5', monitor='val_loss', save_best_only=True)]
    else:
        callbacks = [EarlyStopping(monitor='val_loss', patience=2)]
    return(model,callbacks)

def Train_Steps(params,X_train,X_test,X_val,y_train,y_test,y_val,X_fill,Memory=None):
    epochs = params['epochs']
    np.random.seed(params['iteration'])
    from keras import backend as K
    Mod,callbacks = Dense_Model(params,X_train.shape[1],Memory=Memory)
    batch_size=50#100
    Mod.fit(X_train, # Features
            y_train, # Target vector
            epochs=epochs, # Number of epochs
            callbacks=callbacks, # Early stopping
            verbose=0, # Print description after each epoch
            batch_size=batch_size, # Number of observations per batch
            validation_data=(X_test, y_test)) # Data for evaluation
    Yval = Mod.predict(X_val,batch_size = batch_size)
    MSE = metrics.mean_squared_error(y_val,Yval)
    y_fill = Mod.predict(X_fill,batch_size=batch_size)
    Rsq = metrics.r2_score(y_val,Yval)
    if params['Save']['Model'] == True:
        model_json = Mod.to_json()
        with open(params['Y']+'/'+"Weights/"+str(params['Model'])+".json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        print("Saved model to disk")
        params['Save']['Model'] = False
    return(y_fill,Yval)#,y_val,Rsq)
    #return(MSE,y_fill,Yval,y_val,Rsq)
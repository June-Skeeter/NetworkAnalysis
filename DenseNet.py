import numpy as np 
import pandas as pd
import math
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from functools import partial

def Dense_Model(params,inputs,lr=1e-4):
    import keras
    import keras.backend as K
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.wrappers.scikit_learn import KerasRegressor
    from keras.callbacks import EarlyStopping,ModelCheckpoint,LearningRateScheduler
    import tensorflow as tf

    def Variance_Loss(y_true, y_pred):
        y_pred_std = K.std(y_pred)
        y_sqe = (y_pred-y_true)**2 
        var_loss = K.sum((y_pred_std-y_sqe)**2)/2
        return var_loss

    def Dual_Loss(y_true,y_pred):
        y_pred_std = K.std(y_pred)
        Term1 = (y_pred_std-y_true)**2/y_pred_std
        Term2 = K.log(y_pred_std)
        dual_loss = K.sum((Term1+Term2)**2)/2
        return dual_loss

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = params['Memory']
    session = tf.Session(config=config)
    initializer = keras.initializers.glorot_uniform(seed=params['iteration'])
    LeakyRelu = keras.layers.LeakyReLU(alpha=0.3)
    model = Sequential()
    if params['Loss'] == 'Variance_Loss':
        model.add(Dense(params['N'], input_dim=inputs,activation='relu',kernel_initializer=initializer))#'relu'
        #model.add(LeakyRelu) # - if we want to use leaky relu instead ...
        model.add(Dense(1,activation='elu'))
        # model.add(Activation('exponential'))
        NUM_GPU = 1 # or the number of GPUs available on your machin
        adam = keras.optimizers.Adam(lr = lr)
        gpu_list = []
        for i in range(NUM_GPU): gpu_list.append('gpu(%d)' % i)
        model.compile(loss=Variance_Loss, optimizer='adam')#'mean_absolute_error', optimizer='adam')#,context=gpu_list) # - Add if using MXNET
        patience=5
    else:
        model.add(Dense(params['N'], input_dim=inputs,activation='relu',kernel_initializer=initializer))#'relu'
        #model.add(LeakyRelu) # - if we want to use leaky relu instead ...
        model.add(Dense(1))
        NUM_GPU = 1 # or the number of GPUs available on your machin
        adam = keras.optimizers.Adam(lr = lr)
        gpu_list = []
        for i in range(NUM_GPU): gpu_list.append('gpu(%d)' % i)
        model.compile(loss='mean_absolute_error', optimizer='adam')#,context=gpu_list) # - Add if using MXNET
        patience=2
    
    if params['Save']['Weights'] == True:
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience),
             ModelCheckpoint(filepath=params['Dpath']+params['Y']+'/Weights/'+str(params['Model'])+'_'+str(params['iteration'])+'.h5', monitor='val_loss', save_best_only=True)]
    else:
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience)]
    return(model,callbacks)

def Train_DNN(params,X_train,X_test,X_val,y_train,y_test,y_val):#,X_fill):
    epochs = params['epochs']
    np.random.seed(params['iteration'])
    from keras import backend as K
    Mod,callbacks = Dense_Model(params,X_train.shape[1])
    batch_size=50#100
    Mod.fit(X_train, # Features
            y_train, # Target vector
            epochs=epochs, # Number of epochs
            callbacks=callbacks, # Early stopping
            verbose=0, # Print description after each epoch
            batch_size=batch_size, # Number of observations per batch
            validation_data=(X_test, y_test)) # Data for evaluation
    X_train = np.append(X_train,X_test,axis=0)
    Y_train = Mod.predict(X_train,batch_size = batch_size)
    Y_val = Mod.predict(X_val,batch_size = batch_size)
    # MSE = metrics.mean_squared_error(y_val,Yval)
    # y_fill = Mod.predict(X_fill,batch_size=batch_size)
    # Rsq = metrics.r2_score(y_val,Yval)
    if params['Save']['Model'] == True:
        model_json = Mod.to_json()
        with open(params['Dpath']+params['Y']+'/'+"Weights/"+str(params['Model'])+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        print("Saved model to disk")
        params['Save']['Model'] = False
    return(Y_train,Y_val)#,y_val,Rsq)
    #return(MSE,y_fill,Yval,y_val,Rsq)

def TTV_Split(iteration,params,X,y):#,X_fill):
    # params['seed'] = int(iteration%params['splits_per_mod']/params['splits_per_mod']*100)
    params['iteration'] = iteration#int(iteration/params['splits_per_mod'])
    indicies = np.arange(0,y.shape[0],dtype=int)
    # print(indicies)
    X_train,X_test,y_train,y_test,i_train,i_test=train_test_split(X,y,indicies, test_size=0.1, random_state=params['iteration'])
    X_train,X_val,y_train,y_val,i_train,i_val=train_test_split(X_train,y_train,i_train, test_size=0.11, random_state=params['iteration'])
    Y_train,Y_val=Train_DNN(params,X_train,X_test,X_val,y_train,y_test,y_val)#,X_fill = X_fill)
    y_train = np.append(y_train,y_test,axis=0)
    X_train = np.append(X_train,X_test,axis=0)
    i_train = np.append(i_train,i_test,axis=0)
    return(Y_train,y_train,Y_val,y_val,X_train,X_val,i_train,i_val)

def RunNN(params,X,y,X_fill,yScale,XScale,pool=None):
    params['Memory'] = (math.floor(100/params['proc'])- 5/params['proc']) * .01
    # params['Memory'] = .3
    print(params['Memory'])
    Y_train=[]
    y_train=[]
    X_train=[]
    Y_val=[]
    y_val=[]
    X_val=[]
    i_train=[]
    i_val=[]
    if pool == None:
        for i in range(params['K']):
            results = TTV_Split(i,params,X,y)
            Y_train.append(yScale.inverse_transform(results[0]))#.reshape(-1,1)))
            y_train.append(yScale.inverse_transform(results[1]))#.reshape(-1,1)))
            Y_val.append(yScale.inverse_transform(results[2]))#.reshape(-1,1)))
            y_val.append(yScale.inverse_transform(results[3]))#.reshape(-1,1)))
            X_train.append(XScale.inverse_transform(results[4]))
            X_val.append(XScale.inverse_transform(results[5]))
            i_train.append(results[6])
            i_val.append(results[7])

    else:
        for i,results in enumerate(pool.imap(partial(TTV_Split,params=params,X=X,y=y),range(params['K']))):
            Y_train.append(yScale.inverse_transform(results[0]))#.reshape(-1,1)))
            y_train.append(yScale.inverse_transform(results[1]))#.reshape(-1,1)))
            Y_val.append(yScale.inverse_transform(results[2]))#.reshape(-1,1)))
            y_val.append(yScale.inverse_transform(results[3]))#.reshape(-1,1)))
            X_train.append(XScale.inverse_transform(results[4]))
            X_val.append(XScale.inverse_transform(results[5]))
            i_train.append(results[6])
            i_val.append(results[7])

    Y_train = np.asanyarray(Y_train)#.sum(axis=-1)
    y_train = np.asanyarray(y_train)#.sum(axis=-1)
    Y_val = np.asanyarray(Y_val)#.sum(axis=-1)
    y_val = np.asanyarray(y_val)#.sum(axis=-1)
    X_train = np.asanyarray(X_train)
    X_val = np.asanyarray(X_val)

    # Y_fill = np.asanyarray(Y_fill).sum(axis=-1)
    # Y_eval = np.asanyarray(Y_eval).sum(axis=-1)
    # y_val = np.asanyarray(y_val).sum(axis=-1)
    

    return(Y_train,y_train,X_train,Y_val,y_val,X_val,i_train,i_val)

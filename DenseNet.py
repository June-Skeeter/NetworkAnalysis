import numpy as np 
import pandas as pd
import math
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from functools import partial

def Params(Func,target,MP = True,processes = 3):
    params = {}
    if MP == False:params['proc']=1
    else:params['proc']=processes
    if Func == 'Full':
        K = 30
        splits_per_mod = 4
        N = np.linspace(100,10,8,dtype='int32')
    elif Func == 'Test':
        K = 4
        splits_per_mod = 2
        N = np.linspace(70,10,4,dtype='int32')
    elif Func == 'Single':
        K = 1
        splits_per_mod = 1
        N = np.linspace(70,10,4,dtype='int32')
    N = np.repeat(N,K)
    d = {'N':N.astype(int)}
    Runs = pd.DataFrame(data=d)
    for val in ['RMSE','R2','Mean','Var','Model']:
        Runs[val] = 0
    params['K'] = K
    params['epochs'] = 200
    params['target'] = target
    params['splits_per_mod'] = splits_per_mod
    params['Save'] = {}
    params['Save']['Weights']=True
    params['Save']['Model']=True
    params['Loss']='mean_absolute_error'
    params['Memory']=.3
    params['Validate'] = True
    return(Runs,params)


def Dense_Model(params,inputs,lr=1e-4,patience=2):
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

    def Boot_Loss(y_true,y_pred):
        return (K.sum(K.log(y_pred)+y_true/y_pred)/2)


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
    # LeakyRelu = keras.layers.LeakyReLU(alpha=0.3)
    model = Sequential()
    model.add(Dense(params['N'], input_dim=inputs,activation='relu',kernel_initializer=initializer))#'relu'
    NUM_GPU = 1 # or the number of GPUs available on your machin
    adam = keras.optimizers.Adam(lr = lr)
    gpu_list = []
    for i in range(NUM_GPU): gpu_list.append('gpu(%d)' % i)
    if params['Loss'] == 'Variance_Loss':
        model.add(Dense(1,activation='elu'))
        model.compile(loss=Variance_Loss, optimizer='adam')
    elif params['Loss'] == 'Boot_Loss':
        model.add(Dense(1,activation='elu'))
        model.compile(loss=Boot_Loss, optimizer='adam')
    else:
        # model.add(Dense(params['N'], input_dim=inputs,activation='relu',kernel_initializer=initializer))#'relu'
        #model.add(LeakyRelu) # - if we want to use leaky relu instead ...
        model.add(Dense(1))
        # NUM_GPU = 1 # or the number of GPUs available on your machin
        # adam = keras.optimizers.Adam(lr = lr)
        # gpu_list = []
        # for i in range(NUM_GPU): gpu_list.append('gpu(%d)' % i)
        model.compile(loss='mean_absolute_error', optimizer='adam')#,context=gpu_list) # - Add if using MXNET
        patience=2
    if params['Save']['Weights'] == True:
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience),
             ModelCheckpoint(filepath=params['Spath']+params['Sname']+'.h5', monitor='val_loss', save_best_only=True)]
    else:
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience)]
    return(model,callbacks)

def Train_DNN(params,X_train,y_train,X_test,y_test,X_val=None):#,X_fill):
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
    Y_target = Mod.predict(X_train,batch_size = batch_size)
    if params['Validate']==True:
        Y_val = Mod.predict(X_val,batch_size = batch_size)
        Y_target =np.append(Y_target,Y_val,axis=0)
    if params['Save']['Model'] == True:
        model_json = Mod.to_json()
        with open(params['Spath']+params['Sname']+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        print("Saved model to disk")
    return(Y_target)#,y_val,Rsq)

def TTV_Split(iteration,params,X,y):
    params['iteration'] = iteration
    if params['iteration']>0 and params['Loss'] != 'Variance_Loss':
        params['Save']['Model'] = False
    indicies = np.arange(0,y.shape[0],dtype=int)
    ones = np.ones(y.shape[0],dtype=int)
    # print(indicies)
    X_train,X_test,y_train,y_test,i_train,i_test,ones_train,ones_test=train_test_split(X,y,indicies,ones, test_size=.7, random_state=params['iteration'])#0.2s
    if params['Validate'] == True:
        X_test,X_val,y_test,y_val,i_test,i_val,ones_test,ones_val=train_test_split(X_test,y_test,i_test,ones_test, test_size=.33, random_state=params['iteration'])#0.25s
        Y_hat=Train_DNN(params,X_train,y_train,X_test,y_test,X_val)#,X_fill = X_fill)
        y_true = np.append(y_train,y_test,axis=0)
        X_true = np.append(X_train,X_test,axis=0)
        index = np.append(i_train,i_test,axis=0)
        ones = np.append(ones_train,ones_test)
        y_true = np.append(y_true,y_val,axis=0)
        X_true = np.append(X_true,X_val,axis=0)
        index = np.append(index,i_val,axis=0)
        ones_val *= 0
        ones = np.append(ones,ones_val)
    else:
        Y_hat=Train_DNN(params,X_train,y_train,X_test,y_test)
        y_true = np.append(y_train,y_test,axis=0)
        X_true = np.append(X_train,X_test,axis=0)
        index = np.append(i_train,i_test,axis=0)
        ones = np.append(ones_train,ones_test)
    return(Y_hat,y_true,X_true,index,ones)

def RunNN(params,X,y,yScale,XScale,pool=None):
    params['Memory'] = (math.floor(100/params['proc'])- 5/params['proc']) * .01
    # params['Memory'] = .3
    print(params['Memory'])
    Y_hat=[]
    y_true=[]
    X_true=[]
    index=[]
    ones=[]
    if pool == None:
        for i in range(params['K']):
            results = TTV_Split(i,params,X,y)
            Y_hat.append(yScale.inverse_transform(results[0]))
            y_true.append(yScale.inverse_transform(results[1]))
            X_true.append(XScale.inverse_transform(results[2]))
            index.append(results[3])
            ones.append(results[4])
    else:
        for i,results in enumerate(pool.imap(partial(TTV_Split,params=params,X=X,y=y),range(params['K']))):
            Y_hat.append(yScale.inverse_transform(results[0]))
            y_true.append(yScale.inverse_transform(results[1]))
            X_true.append(XScale.inverse_transform(results[2]))
            index.append(results[3])
            ones.append(results[4])

    Y_hat = np.asanyarray(Y_hat)
    y_true = np.asanyarray(y_true)
    X_true = np.asanyarray(X_true)
    index = np.asanyarray(index)
    ones = np.asanyarray(ones)


    return(Y_hat,y_true,X_true,index,ones)

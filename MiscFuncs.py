import numpy as np
import pandas as pd
from scipy import stats
from keras.models import model_from_json

def Load_Model(params):
    print(params['Dpath']+'/'+params['Y']+'/Weights/'+params['Model']+'.json')
    json_file = open(params['Dpath']+'/'+params['Y']+'/Weights/'+params['Model']+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return(loaded_model)

def Load_Weights(loaded_model,i,s,params):
    loaded_model.load_weights(params['Dpath']+'/'+params['Y']+'/Weights/'+params['Model']+'_'+str(i)+'_'+str(s)+'.h5')
    loaded_model.compile(loss='mean_squared_error', optimizer='adam')
    return(loaded_model)


def PI(X,Xh,MSE):
    Xht = np.transpose(Xh)
    Xt = np.transpose(X)
    Xdot = np.dot(Xt,X)
    Xinv=np.linalg.inv(Xdot)
    SE = MSE*(np.dot(np.dot(Xht,Xinv),Xh))
    print(SE)
    PI = (SE**2+MSE)**.5*stats.t.ppf(1-0.025,X.shape[0]-X.shape[1])
    return(PI)


def Map(X,dx,params,ax,color,label,RST,Derivs = True):
    EmptyModel = Load_Model(params)
    results = []
    for i,s in zip(params['Runs']['iteration'],params['Runs']['seed']):
        Model = Load_Weights(EmptyModel,i,s,params) 
        Yfill=RST.YScaled.inverse_transform(Model.predict(X).reshape(-1,1))
        results.append(Yfill)
    y = np.asanyarray(results).mean(axis=0)    
    X = RST.XScaled.inverse_transform(X)
    X_og = RST.XScaled.inverse_transform(RST.X)
    data = pd.DataFrame(X,columns=params['Vars'])
    data['Pred']=y
    if Derivs == True:
        ax[0].plot(data[dx],data['Pred'],color=color,label=label)
        data = data.sort_values(by=dx)
        data = data.groupby(dx).mean().reset_index()
        data['d'+params['Y']+'/d'+dx]=(data['Pred'].diff()/data[dx].diff())
        ax[1].plot(data[dx],data['d'+params['Y']+'/d'+dx],(),color=color)
    else:
        ax.plot(data.index,data['Pred'],color=color,label=label)
    print(params['Runs']['RMSE'].quantile(.025))
    print(params['Runs']['RMSE'].quantile(.975))
    pred_int = PI(X_og,X.mean(axis=0),params['Runs']['RMSE'].mean()**2)
    return(ax,data['Pred'].mean(),pred_int)
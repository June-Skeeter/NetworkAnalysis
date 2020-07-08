# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:24:38 2020

@author: wesle
"""

import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from multiprocessing import Pool
from functools import partial
from sklearn import metrics

## Personal Modules
import ReadStandardTimeFill as RSTF
import importlib
import DenseNet as Dense
import MiscFuncs as MF
importlib.reload(Dense)
importlib.reload(RSTF)
importlib.reload(MF)

# %matplotlib inline

#%matplotlib notebook
#%config IPCompleter.greedy=True

from scipy.optimize import minimize, curve_fit
from scipy.stats import norm
from sklearn.externals import joblib
from matplotlib import cm

from scipy import stats

from ipywidgets import FloatProgress, HTML
from IPython.display import display, clear_output
import os  
import shutil
from keras import backend as K
try:pool.close()
except:pass

import json


def Test(params,X,y,YScaled,XScaled,pool):
    return(np.random.rand(params['K']))


def ModSelect(Scope,Site):
    if Site == 'Illisarvik':
        if Scope == 'Full':
            Model = ['PPFD_Avg','AirTC_Avg','VPD','wind_speed',
                    'Temp','VWC','Sedge','Shrub','Grass','Upland',
                    'HR','DOY']
        if Scope == 'Test':
            Model = ['PPFD_Avg','VPD','wind_speed','VWC']
    if Site == 'FishIsland':
        BaseFactors = []
        if Scope == 'Full':
            Model=['H','AirTC_Avg','VPD','NR_Wm2_Avg','PPFD_Avg','Table_1','Delta_Table_1',
                   'VWC_2','Delta_VWC_2','Temp_2_5_1','Temp_5_1','Temp_15_1','Active_Layer_1',
                   'u*','wind_speed']#,'air_pressure','Delta_air_pressure']
        if Scope == 'Test':
            Model=['H','AirTC_Avg','RH_Samp','VPD','NR_Wm2_Avg','PPFD_Avg','Table_1','Delta_Table_1',
            'VWC_2','VWC_1','Temp_2_5_1','Temp_5_1','Temp_15_1','Delta_Temp_1',
            'Active_Layer_1','u*','wind_speed','air_pressure']
    return(Model)

def Stats(mse,se,r2,j,params,i=0):
    df = pd.DataFrame(index = [str(j)+'_'+str(i)],
                      data={'Model':[params['Model']],'MSE':[mse],'Size':j,'Number':i,'HiddenNodes':params['N'],
                            'SE':[se],'r2':[r2],'Performance':0,'K':[params['K']]})
    return(df)

def t(p,n):
    alpha = 1-p
    df = n-1
    return(stats.t.ppf(alpha,df))
MP = False
MP = True
# Scope = 'Full'
Scope = 'Test'
if MP == True:
    processes = 3
else:
    processes = 1
# memory = .95/processes
    
cwd = os.getcwd()
Site='FishIsland'
# Site = 'Illisarvik'
alpha = .05

pd.set_option('max_colwidth',200)
#def Display (tar,prog1=None,prog2=None,MdLs=None,MdL=None):
#    clear_output()
#    display(tar)
#    if prog1!=None:
#        display(prog1)
#    if prog2!=None:
#        display(prog2)
#    if MdLs!=None:
#        display(MdLs)
#    if MdL!=None:
#        display(MdL)
#    
#tar = HTML(
#            value=" ",
#            placeholder='Target: ',
#            description='Target: ',
#        )
#kwt = HTML(
#        value=str(0),
#        placeholder='Quit Score: ',
#        description='Quit Score: ',
#        )


# depth = 15
Time = time.time()
AllRes={}
if __name__ == '__main__':
    for target in ['fco2','fch4']:
        AllRes[target]={}
        AllRes[target]['Results'] = {}
        AllRes[target]['Derivatives'] = {}
        AllRes[target]['SSQ'] ={}
        AllRes[target]['X'] = {}
        AllRes[target]['Y_hat'] = {}
        AllRes[target]['Y_true'] = {}
        AllRes[target]['Outputs'] = {}
        AllRes[target]['Factors'] = []
        Rm = []
        Input=ModSelect(Scope,Site)
#         Input = ['u*','wind_speed','PPFD_Avg','NR_Wm2_Avg','Temp_15_1','Temp_2_5_1','Table_1','Delta_Table_1','Active_Layer_1']
#         Input = ['AirTC_Avg','VPD','PPFD_Avg','Table_1','VWC_2','Temp_5_1','Temp_15_1',
#                  'Active_Layer_1','u*','wind_speed']
#         Models = [['H','AirTC_Avg','VPD','NR_Wm2_Avg','PPFD_Avg','Table_1','Delta_Table_1','VWC_2','Delta_VWC_2',
#                    'Temp_2_5_1','Temp_5_1','Temp_15_1','Active_Layer_1','u*','wind_speed',
#                     'North','East','air_pressure'],
#                  ['H','AirTC_Avg','VPD','NR_Wm2_Avg','PPFD_Avg','Table_1','Delta_Table_1','VWC_2','Delta_VWC_2',
#                    'Temp_2_5_1','Temp_5_1','Temp_15_1','Active_Layer_1','u*','wind_speed',
#                     'air_pressure'],
#                  ['H','AirTC_Avg','VPD','NR_Wm2_Avg','PPFD_Avg','Table_1','VWC_2',
#                    'Temp_2_5_1','Temp_5_1','Temp_15_1','Active_Layer_1','u*','wind_speed']]
        start = len(Input)
        IpKey=np.arange(0,start)
        IpDict={'Factors':Input.copy(),'Key':IpKey}
        try:shutil.rmtree(cwd+'/'+Site+'/'+target+'/')
        except:pass
        os.mkdir(cwd+'/'+Site+'/'+target+'/')  
#        tar.value=target
#        prog2 = FloatProgress(min=0, max=100,description='Bootstrapping:')
#        MdL = HTML(value=" ",placeholder='Testing: ',description='Testing: ')
#        Display (tar,prog2=prog2,MdL=MdL)
        Continue = True
        first = 1
        num = 0
#         for Input in Models:
        while Continue == True:
            Time2 = time.time()
            j = len(Input)
            params = Dense.Params(Scope,target,MP=MP)
            params['Dpath'] = cwd+'/'+Site+'/'
            params['Spath'] = params['Dpath']+target+'/'+str(j)+'_'+str(num)+'/'
            try:os.mkdir(params['Spath'])
            except:pass
            params['Sname'] = 'Y_'
            params['Inputs'] = Input
#            MdL.value='N = '+str(j)+':  '+str(params['Inputs'])
#             Display (tar,prog1,prog2,MdLs,MdL)
            params['Model'] = '+'.join(params['Inputs'])
            RST = RSTF.ReadStandardTimeFill(params,'AllData.csv')#,resample='2H')
            if target == 'ER':
                RST.Master = RST.Master.loc[RST.Master['fco2']>0]
            RST.Scale(params['target'],params['Inputs']) 
            y = RST.y*1.0
            X = RST.X*1.0
            Ni = len(Input)
            Ns = y.shape[0]
            No = 1
            a = 2
            params['N']=int(Ns/(a*(Ni+No)))
            # Rule by Maier et al. 1998
#             A = int(2*Ni+No)
#             B = int(2*Ns/((Ni+No)))
#             if A<B:params['N']=A
#             else:params['N']=B
                
            params['Memory'] = (math.floor(100/params['proc'])- 5/params['proc']) * .01
            Y_hat=[]
            y_true=[]
            X_true=[]
            index=[]
            ones=[]
#            prog2.value=0
            Avs = []
            Derivatives = []
            Outputs=[]
#             History = []
            print(params)
            if MP == False:
                for k in range(params['K']):
                    results = Dense.Bootstrap(k,params=params,X=X,y=y)
                    Y_hat.append(RST.YScaled.inverse_transform(results[0]))
                    y_true.append(RST.YScaled.inverse_transform(results[1]))
                    X_true.append(RST.XScaled.inverse_transform(results[2]))
                    ones.append(results[3])
#                    prog2.value=(k+1)/params['K']*100
                    Avs.append(results[4])
                    Derivatives.append(results[5])
                    Outputs = results[6]
            else:
                pool = Pool(processes=processes,maxtasksperchild=75)
#                 R=[]
#                 R = (pool.apply_async(partial(Dense.Bootstrap,params=params,X=X,y=y), range(params['K'])))
# #                 partial(Dense.Bootstrap,params=params,X=X,y=y),range(params['K'])
#                 print(R)
#                 for results in (R):
#                     print(results.get(timeout=1))
                for k,results in enumerate(pool.imap_unordered(partial(Dense.Bootstrap,params=params,X=X,y=y),range(params['K']))):
                    print(k)
                    Y_hat.append(RST.YScaled.inverse_transform(results[0]))
                    y_true.append(RST.YScaled.inverse_transform(results[1]))
                    X_true.append(RST.XScaled.inverse_transform(results[2]))
                    ones.append(results[3])
#                    prog2.value=(k+1)/params['K']*100
                    Avs.append(results[4])
                    Derivatives.append(results[5])
                    Outputs.append(results[6])
                pool.close()
            Y_hat = np.squeeze(np.asanyarray(Y_hat))
            y_true = np.squeeze(np.asanyarray(y_true))
            X_true = np.asanyarray(X_true)
            ones = np.asanyarray(ones)
            params['Memory'] = .95
            results = Dense.Sort_outputs(0,params=params,Y_hat=Y_hat,y_true=y_true,X_true=X_true,ones=ones)
            mse,se,r2 = results
            Level = Stats(mse,se,r2,j,params,i=num)
            Level.to_csv('Temp.csv')
#             prog1.value=start-j#+i/len(Inputs)
            Level['Thresh'] = Level['MSE']+Level['SE']
            Min = Level.loc[Level['MSE']==Level['MSE'].min()]
            if first == 1:
                Records = Level
            else:
                Records = Records.append(Level)
            Av = np.array(Avs).mean(axis=0)
            Drv = np.array(Derivatives)#.mean(axis=0)
            Av = (Drv.mean(axis=0)**2).sum(axis=1)
            SE = (np.array(Avs).T/np.array(Avs).sum(axis=1)).T
            SE = SE.std(axis=0)**.5/(params['K']**.5)
            if first==2:
                IpDict['RC: '+str(j) +'_'+ str(num)]=np.zeros(Results.shape[0])#Av/Av.sum()
                IpDict['SE: '+str(j) +'_'+ str(num)]=np.zeros(Results.shape[0])#Av/Av.sum()
            else:
                first = 2
            Results = pd.DataFrame(data=IpDict)
            for I,A,S in zip(Input,Av,SE):
                Results.loc[Results['Factors']==I,'RC: '+str(j)+'_'+ str(num)]=A
                Results.loc[Results['Factors']==I,'SE: '+str(j)+'_'+ str(num)]=S
            Results['RC: '+str(j)+'_'+ str(num)]=Results['RC: '+str(j)+'_'+ str(num)]/Results['RC: '+str(j)+'_'+ str(num)].sum()
            IpDict['RC: '+str(j)+'_'+ str(num)]=Results['RC: '+str(j)+'_'+ str(num)].values
            IpDict['SE: '+str(j)+'_'+ str(num)]=Results['SE: '+str(j)+'_'+ str(num)].values
            Results = Results.sort_values('RC: '+str(j)+'_'+ str(num))
            Rm.append(Results.Factors.values[num])
            print(Rm,Results.Factors.values[num])
            Input.remove(Rm[-1])
            Results = Results.sort_values('Key')
            Min = Records.loc[Records['MSE']==Records['MSE'].min()]
            if Min['MSE'].values[0]+Min['SE'].values[0]<=Level['MSE'].values[0]:
#             if num > 2:
                Continue = False
            AllRes[target]['Records']=Records
            AllRes[target]['Results'] = Results
            AllRes[target]['Derivatives'][j]=Drv
            AllRes[target]['SSQ'][j]=Avs
            AllRes[target]['X'][j] = X_true[0]
            AllRes[target]['Outputs'][j] = Outputs
            AllRes[target]['Y_hat'][j] = Y_hat
            AllRes[target]['Y_true'][j] = y_true
            AllRes[target]['Factors'].append(j)
            AllRes[target]['Removed']=Rm
            Records.to_csv('C:\\Users\\wesle\\NetworkAnalysis\\FishIsland/'+target+'_Test.csv')
            num +=1
            print('Total Runtime: ',time.time()-Time)
            print('Training Time: ', time.time()-Time2)
#             with open(target+'dict.txt','w') as file:
#                 file.write(json.dumps(AllRes))
#             j+=1
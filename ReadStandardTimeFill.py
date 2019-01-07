import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

class ReadStandardTimeFill:
    def __init__(self,params,Name,CombineKeys=[],Conversions=[1e-6 * 44.0095 *3600,1e-3 * 16.04246 *3600],resample=None):
        self.Master = pd.read_csv(params['Dpath']+Name,delimiter = ',',header = 0,na_values = -9999)
        self.Master = self.Master.set_index(pd.DatetimeIndex(pd.to_datetime(self.Master['datetime'])))
        self.Master['DOY'] = self.Master.index.dayofyear*1.0
        self.Master['HR'] = self.Master.index.hour*1.0
        self.Master['fco2'] *= Conversions[0]
        self.Master['fch4'] *= Conversions[1]
        self.params=params
        if len(CombineKeys) >0:
            for i in range(0,len(CombineKeys),2):
                self.Master[CombineKeys[i]]=self.Master[CombineKeys[i+1]].sum(axis=1)
        self.TimeSteps=0
        if resample != None:
            self.Master=self.Master.resample(resample).mean()
        
    def Scale(self,y_var,X_vars,ScalePath = None,Project=False):
        self.y_var = y_var
        if Project == False:
            self.Data = self.Master[np.isfinite(self.Master[y_var])]
            self.y = self.Data[y_var].values
        self.Data = self.Data.interpolate().bfill()
        self.Data = self.Data.interpolate().ffill()
        if ScalePath is None:
            YStandard = StandardScaler()
            self.YScaled = YStandard.fit(self.y.reshape(-1, 1))
            Yscale = self.YScaled.transform(self.y.reshape(-1, 1))
            self.y = np.ndarray.flatten(Yscale)
        else:
            self.YScaled = joblib.load(ScalePath+'Y_scaler.save') 
        self.Ytru = self.YScaled.inverse_transform(self.y.reshape(-1,1))
        X = self.Data[X_vars]
        self.input_shape = len(X_vars)
        if ScalePath is None:
            XStandard = StandardScaler()
            self.XScaled = XStandard.fit(X)
        else:
            self.XScaled = joblib.load(ScalePath+'X_scaler.save') 
        # self.XScaled = XStandard.fit(X)
        self.X = self.XScaled.transform(X)
        Filling = self.Master[X_vars]
        Filling = Filling.interpolate().bfill()
        Filling = Filling.interpolate().ffill()
        #XStandard = StandardScaler()
        #self.XFillScaled= XStandard.fit(Filling)
        self.X_fill = self.XScaled.transform(Filling)
        if ScalePath is None:
            scaler_filename = "Y_scaler.save"
            joblib.dump(self.YScaled, self.params['Spath']+scaler_filename) 
            scaler_filename = "X_scaler.save"
            joblib.dump(self.XScaled, self.params['Spath']+scaler_filename)

        
    def TimeShape(self,rolls):
        X1 = self.X
        Xf = self.X_fill
        self.X_time = np.zeros(shape = (X1.shape[0],rolls+1,X1.shape[1]))
        self.X_time[:,0,:] = X1
        self.X_ftime = np.zeros(shape = (Xf.shape[0],rolls+1,Xf.shape[1]))
        self.X_ftime[:,0,:] = Xf
        if rolls > 0:
            for roll in range(0,rolls):
                X2 = np.roll(X1,(roll+1),axis=0)
                X2f = np.roll(Xf,(roll+1),axis=0)
                self.X_time[:,roll+1,:] = X2
                self.X_ftime[:,roll+1,:] = Xf
        self.X_time = self.X_time[rolls+1:,:,:]
        self.X_ftime = self.X_ftime[rolls+1:,:,:]
        self.y_time = self.y[rolls+1:]
        self.y_ftime = self.y[rolls+1:]
        self.TimeSteps = rolls+1
        
    def Fill(self,Y_Pred,Name):
        Y_fill = self.YScaled.inverse_transform(Y_Pred.reshape(-1,1))
        if self.TimeSteps>0:
            nanz = np.zeros(shape=(self.TimeSteps,1))
            nanz[:,:] = np.nan
            Y_Pred = np.concatenate((nanz,Y_fill),axis=0).reshape(-1,1)
        else:
            Y_Pred = Y_fill
        self.Master['TempFill'] = Y_Pred
        self.Master[Name] = self.Master[self.y_var].fillna(self.Master['TempFill'])
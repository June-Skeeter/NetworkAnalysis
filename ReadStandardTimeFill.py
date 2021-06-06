
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
# from sklearn.externals import joblib
import joblib
import os

class ReadStandardTimeFill:#[1e-6 * 44.0095 *3600,1e-3 * 16.04246 *3600]
    def __init__(self,params,Name,CombineKeys=[],Conversions=[1,1e3],resample=None,Const=False,FPVars=None):
        self.Master = pd.read_csv(params['Dpath']+Name,delimiter = ',',header = 0,na_values = -9999)
        # self.Master.loc[self.Master['VWC']<.5,'VWC']=np.nan
        self.Master = self.Master.set_index(pd.DatetimeIndex(pd.to_datetime(self.Master['datetime'])))
        self.Master['DOY2'] = self.Master.index.dayofyear*1.0
        self.Master['HR'] = self.Master.index.hour*1.0
        self.Master['fco2'] *= Conversions[0]
        # self.Master['ER'] = np.nan
        self.Master.loc[self.Master['Daytime']==0,'ER']==self.Master.loc[self.Master['Daytime']==0,'co2_flux']
        self.Master.loc[self.Master['Daytime']==1,'ER']= np.nan
        self.Master['ER'] *= Conversions[0]
        self.Master['fch4'] *= Conversions[1]

        if Const==True:
            self.Master['Constant']=1
        self.params=params
        if len(CombineKeys) >0:
            for i in range(0,len(CombineKeys),2):
                self.Master[CombineKeys[i]]=self.Master[CombineKeys[i+1]].sum(axis=1)
        self.TimeSteps=0
        if resample != None:
            self.Master=self.Master.resample(resample).mean()
        if FPVars != None:
            self.FPFill(FPVars=FPVars[0],bad=FPVars[1])

    def FPFill(self,FPVars,bad):
        self.Master['WindBins'] = (self.Master['wind_dir']/10).round()
        Mn = self.Master[FPVars].groupby(FPVars[-1]).mean()
        for i, row in self.Master.loc[np.isnan(self.Master[FPVars[0]])==True].iterrows():
            if row[FPVars[-1]]!=bad and np.isnan(row[FPVars[-1]])==False:
                self.Master.loc[self.Master.index==i,FPVars[:-2]] = Mn.loc[Mn.index==row[FPVars[-1]],FPVars[:-2]].values[0]
                self.Master.loc[self.Master.index==i,FPVars[-2]] = 1 - self.Master.loc[self.Master.index==i,FPVars[:-2]].values[0].sum()
                # print(self.Master.loc[self.Master.index==i,FPVars[-2]],(1 - self.Master.loc[self.Master.index==i,FPVars[:-2]].sum()))
        # Mn = self.Master[FPVars].groupby(FPVars[-1]).count()
        
    def Scale(self,y_var,X_vars,ScalePath = None,Project=False,fillTarget=None):
        self.y_var = y_var
        if Project == False:
            self.Data = self.Master[np.isfinite(self.Master[y_var])]
        self.y = self.Data[y_var].values
        self.Ytru = self.y+0.0
        self.Data = self.Data.interpolate().bfill()
        self.Data = self.Data.interpolate().ffill()
        if ScalePath is None:
            YStandard = StandardScaler()
            # YStandard = MinMaxScaler(feature_range=(.3, .7))
            self.YScaled = YStandard.fit(self.y.reshape(-1, 1))
            Yscale = self.YScaled.transform(self.y.reshape(-1, 1))
            self.y = np.ndarray.flatten(Yscale)
        else:
            self.YScaled = joblib.load(ScalePath+'Y_scaler.save') 
            self.YvarScaled = joblib.load(ScalePath+'YVar_scaler.save')
            Yscale = self.YScaled.transform(self.y.reshape(-1, 1))
            self.y = np.ndarray.flatten(Yscale)
        X = self.Data[X_vars]
        self.input_shape = len(X_vars)
        if ScalePath is None:
            XStandard = StandardScaler()
            # XStandard = MinMaxScaler()
            self.XScaled = XStandard.fit(X)
        else:
            self.XScaled = joblib.load(ScalePath+'X_scaler.save') 
        # self.XScaled = XStandard.fit(X)
        self.X = self.XScaled.transform(X)
        if fillTarget is None:
            Filling = self.Master[X_vars]
        else:
            Filling = fillTarget[X_vars]

        Filling = Filling.interpolate().bfill()
        Filling = Filling.interpolate().ffill()
        self.X_fill = self.XScaled.transform(Filling)
        if ScalePath is None:
            try:
                os.mkdir(self.params['Spath'])
            except:
                pass
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
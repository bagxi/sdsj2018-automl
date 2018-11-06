import datetime
import time
import pandas as pd
import numpy as np
import lightgbm as lgbm
import xgboost as xgb
from tqdm import tqdm
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from linear_model import linear_model

class modelcomplex:
    def __init__(self):
        self.xgb = None
        self.lgb = None
        self.stack = None        
    
    def predict(self,df):
        if self.xgb is not None:
            xgb_preds = self.xgb.predict(xgb.DMatrix(df))
            result = xgb_preds
        if self.lgb is not None:
            lgb_preds = self.lgb.predict(df,num_iteration = self.lgb.best_iteration)
            result = lgb_preds
        if self.stack is not None:
            if isinstance(self.stack,Ridge):
                result = self.stack.predict(np.transpose([xgb_preds,lgb_preds]))
            else:
                result = self.stack.predict_proba(np.transpose([xgb_preds,lgb_preds]))[:,1]
        return result

class automl:
    def __init__(self,mode,time_limit=5*60):
        self.mode = mode
        self.tl = time_limit        
        self.ds_save_size = 300 * 1024 * 1024        
        self.alpha_power = 0.05
        self.memory_border = 2500.
        self.big_file = False
        self.small_file = False
        self.hard_rules = list()
        self.shifted_match = None
        self.shifted = []
        
        self.ste = {}
        
    def train(self,input_file):
        start_time = time.time()
        
        df_x,df_y = self.load_data(input_file)        
        if df_x.memory_usage().sum() < self.ds_save_size:
            self.train_ds = (df_x,df_y)
        
        if self.shifted_match is not None:
            mask = ~df_x[self.shifted_match].isnull()
            print(df_y.loc[mask].sum(), df_x.loc[mask][self.shifted_match].sum())
            if df_y.loc[mask].sum() == df_x.loc[mask][self.shifted_match].sum():
                df_x[self.shifted_match] = 0
            else:
                self.shifted_match = None
        train_start_time = time.time()
        if self.small_file and self.mode == 'regression':
            self.model = linear_model(mode='regression', random_seed=82)
            self.model.train(df_x, df_y, ds=self.dt_col, train_tl=self.tl-(train_start_time - start_time)-5)
        else:
            self.train_lgbm(df_x,df_y,train_tl = self.tl-(train_start_time - start_time)-5)
        if self.mode == 'regression' and df_y.min() >= 0:
            self.clip_zero = True
        self.train_time = time.time() - train_start_time
        
    def predict(self,input_file):
        start_time = time.time()
        df,result = self.load_data(input_file,False)
        #if we can check and retrain model
        '''
        if self.train_time*1.5 < self.tl and 'train_ds' in self.__dict__:
            if self.check_column_overlap(df):
                print("retrain model")
                df_x, df_y = self.train_ds
                df_x = df_x[self.used_columns]
                self.train_lgbm(df_x,df_y,(self.tl-(time.time()-start_time))/2)
                df = df[self.used_columns]
        '''
        predict_start_time = time.time()
        result['prediction'] = self.model.predict(df)
        '''
        try:
            mask,ref_targets = self.hard_rules_map(df)
            #result.loc[mask,'prediction'] = ref_targets[mask]
        except:
            result['prediction'] = 0
        '''
        if self.mode == 'classification':
            epsilon = 10e-5
            result['prediction'] = np.clip(result['prediction'].values,epsilon,1-epsilon)
        if 'clip_zero' in self.__dict__:
            result['prediction'] = np.clip(result['prediction'].values,0,result['prediction'].max())
        #if self.shifted_match is not None:
        #    mask = ~df[self.shifted_match].isnull()
        #    result.loc[mask,'prediction'] = df.loc[mask,self.shifted_match]
        
        return result
    
    def train_lgbm(self, df_x, df_y,train_tl = 1800):
        train_start_time = time.time()
        lgbm_param = {'learning_rate' :0.03,'max_depth':4,
                     'min_child_weight':16, 'application':'regression_l2',
                     'subsample':0.8, 'colsample_bytree': 0.8,
                     'reg_alpha':0.1,'data_random_seed':42,'metric':'rmse',                     
                     'max_bin':255,'reg_lambda':0.1,
                     'objective':'reg:linear','eval_metric':'rmse'
        }
        if df_x.shape[0] < 10000:
            lgbm_param['learning_rate'] = 0.01
        if self.mode == 'classification':
            lgbm_param['application'] = 'binary'
            lgbm_param['objective'] = 'binary:logistic'
            lgbm_param['metric'] = 'binary_logloss'
            lgbm_param['eval_metric'] = 'logloss'
        X_train, X_test, y_train, y_test = train_test_split(df_x,df_y, test_size=0.1, random_state=42)
        self.model = modelcomplex()
        if 'train_ds' in self.__dict__:
            train_ds = xgb.DMatrix(X_train, label=y_train)
            val_ds = xgb.DMatrix(X_test, label=y_test)
            iters = 1
            self.model.xgb = xgb.train(lgbm_param, train_ds, num_boost_round = 700, early_stopping_rounds = 20,evals = ((val_ds,"val"),),verbose_eval=True)
            while self.model.xgb.best_iteration > 700-20 and (time.time()-train_start_time)/iters*(iters+1) < train_tl:
                self.model.xgb = xgb.train(lgbm_param, train_ds, num_boost_round = 700, early_stopping_rounds = 20,evals = ((val_ds,"val"),),verbose_eval=True, xgb_model=self.model.xgb)
                iters+=1        
        if time.time()-train_start_time < train_tl*2/3:
            #if self.model.xgb is None:
            del lgbm_param['objective']            
            train_ds = lgbm.Dataset(X_train, label=y_train)
            val_ds = lgbm.Dataset(X_test, label=y_test)
            self.model.lgb = lgbm.train(lgbm_param, train_ds, num_boost_round = 5000, early_stopping_rounds = 20,valid_sets = val_ds,verbose_eval=True)
        #simple stacking
        if self.model.lgb is not None and self.model.xgb is not None:
            stacked_preds = np.transpose([self.model.xgb.predict(xgb.DMatrix(X_test)),self.model.lgb.predict(X_test)])
            if self.mode == 'classification':
                self.model.stack = LogisticRegression()
            else:
                self.model.stack = Ridge()
            self.model.stack.fit(stacked_preds,y_test)
            
    def hard_rules_map(self,df):
        mask = np.array([False]*df.shape[0])
        ref_targets = np.zeros(df.shape[0])
        for c,value,ref_target in self.hard_rules:
            if c not in df:continue
            cur_mask = df[c]==value
            if self.mode == 'regression':
                mask = mask|cur_mask
                ref_targets[cur_mask] = ref_target
        return mask,ref_targets
        
    def check_column_overlap(self,test_df):
        result = False
        df_x, df_y = self.train_ds
        train_min = df_x.min()
        train_max = df_x.max()
        result_cols = list()
        test_size = test_df.shape[0]
        for c in self.used_columns:
            if c == self.shifted_match:
                result_cols.append(c)
                continue            
            overlap = test_df[(test_df[c] >= train_min[c])&(test_df[c] <= train_max[c])].shape[0]            
            if overlap < test_size*0.6:
                result = True
            else:
                result_cols.append(c)
        if result and len(result_cols) > 1:
            self.used_columns = result_cols
        return result
    
    def load_data(self,input_file,train=True):
        #preload data
        rowcount = 0
        with open(input_file) as f:
            l = f.readline()
            cols = l.split(",")
            while l != "":
                l = f.readline()
                rowcount+=1
        est_memory = rowcount*len(cols)*8/1024/1024
        dtypes = None
        print(est_memory)
        if est_memory > self.memory_border and train:
            dtypes = {}
            for c in cols:
                if c.startswith('number'):
                    dtypes[c] = np.float32
            est_memory = 4*rowcount/1024/1024*(len(cols)*2-len(dtypes))
            rowcount = int(self.memory_border/est_memory*rowcount)
            print(est_memory,rowcount)
            self.big_file = True
        if rowcount < 1000:
            self.small_file = True
        #df = pd.read_csv(input_file,nrows = 10,low_memory=False)
        datecols = list(c for c in cols if c.startswith('datetime'))
        #load data
        df = pd.read_csv(input_file,nrows = rowcount,low_memory=False,parse_dates=datecols,dtype = dtypes)        
        #drop constants
        if train:self.constant_columns = [col_name for col_name in df.columns if df[col_name].nunique() <= 1]
        df.drop(self.constant_columns, axis=1, inplace=True)
        #save datetime column for linear model
        if self.small_file and train: self.dt_col = df.get('datetime_0')
        
        #add time shifted columns
        if 'datetime_0' in df and 'id_0' in df:
            merge = None
            if train:
                self.shifted = list()
                df2 = df.copy().drop(['target'],axis = 1)
                df2['datetime_0'] = df2['datetime_0'] + datetime.timedelta(days=-1)
                merge = pd.merge(df,df2,on=['datetime_0','id_0'],how = 'left')
                corr_matrix = merge[list(('target',))+list(c for c in merge.columns.values if c.startswith('number') and c.endswith('y'))].corr()[['target']]
                corr_matrix = corr_matrix[corr_matrix.index != 'target']
                corr_matrix['target_x'] = corr_matrix['target'].abs()
                corr_matrix = corr_matrix.sort_values('target',ascending = False)
                shifted = corr_matrix[corr_matrix['target'] > 0.25].sort_values('target',ascending=False)
                if shifted.shape[0] > 0 and shifted.values[0][0] == 1:
                    self.shifted_match = shifted.index[0]                    
                self.shifted = shifted.index.values
            if len(self.shifted) > 0:
                if merge is None:
                    df2 = df.copy()
                    df2['datetime_0'] = df2['datetime_0'] + datetime.timedelta(days=-1)
                    merge = pd.merge(df,df2,on=['datetime_0','id_0'],how = 'left')
                df = pd.concat([df, merge[self.shifted]],axis = 1)
        #process dates
        df = self.transform_datetime_features(df,train)
        #search hard rules
        if train and df.memory_usage().sum() < self.ds_save_size:            
            for c in df.columns:
                groups = df.groupby(c)[['target']].mean()
                if groups.shape[0] > 2:continue
                for g in groups.iterrows():
                    value = g[0]                    
                    ref_target = g[1]['target']
                    if ref_target == 0 or ref_target == 1:
                        self.hard_rules.append((c,value,ref_target))
        #smoothed target encode
        if train:
            self.global_mean = df['target'].mean()
            self.alpha = df.shape[0]*self.alpha_power
            for c in df.columns:
                if c.startswith("string") or c.startswith("id"):
                    counts = pd.DataFrame(pd.concat([df.groupby(c)['target'].count(),df.groupby(c)['target'].mean()],axis = 1))
                    counts.columns = ['count','mean']
                    counts = counts[counts['count']>=10]
                    counts['sl'] = (counts['mean']*counts['count'] + self.global_mean*self.alpha)/(counts['count']+self.alpha)                    
                    self.ste[c] = counts['sl'].to_dict()
                if c.startswith("number") and not self.small_file:
                    nunique = df[c].nunique()
                    if nunique > 31 or nunique < 3: continue
                    counts = pd.DataFrame(pd.concat([df.groupby(c)['target'].count(),df.groupby(c)['target'].mean()],axis = 1))
                    counts.columns = ['count','mean']
                    counts = counts[counts['count']>=10]
                    counts['sl'] = (counts['mean']*counts['count'] + self.global_mean*self.alpha)/(counts['count']+self.alpha)
                    self.ste[c] = counts['sl'].to_dict()
        for c in df.columns:
            if c in self.ste:
                #new_values = set(df[~df[c].isnull()][c].values)
                #for v in new_values:
                #    if v not in self.ste[c]:
                #        self.ste[c][f] = self.global_mean
                df[c] = df[c].map(self.ste[c]).fillna(self.global_mean)
                
        if train:
            self.used_columns = [col_name for col_name in df.columns if col_name.startswith(('number','string','id'))]            
            x,y = (df[self.used_columns],df.target)
        else:
            x,y = (df[self.used_columns],df[['line_id']])
        if self.big_file:x = x.astype(np.float32)
        return x,y
        
    def transform_datetime_features(self,df,train):
        if train:
            self.dt_process = {}
            self.datetime_columns = [col_name for col_name in df.columns if col_name.startswith('datetime')]
            
        for col_name in self.datetime_columns:
            min_date,max_date = (df[col_name].min(),df[col_name].max())
            df['number_daydiff_{}'.format(col_name)] = (df[col_name]-datetime.datetime(2018,8,1)).dt.days
            if not(min_date < datetime.datetime(2010,1,1) or max_date > datetime.datetime(2018,8,1)):
                df['number_day_{}'.format(col_name)] = df[col_name].dt.day
                df['number_weekday_{}'.format(col_name)] = df[col_name].dt.weekday
                year = df[col_name].dt.year
                if train:self.dt_process[col_name+"_year"] = len(set(year[~year.isnull()].values)) > 1                    
                if self.dt_process[col_name+"_year"]:df['number_year_{}'.format(col_name)] = year
                    
                month = df[col_name].dt.month
                if train:self.dt_process[col_name+"_month"] = len(set(month[~month.isnull()].values)) == 12
                if self.dt_process[col_name+"_month"]:df['number_month_{}'.format(col_name)] = month
                    
                hours = df[col_name].dt.hour
                if train:self.dt_process[col_name+"_hour"] = len(set(hours[~hours.isnull()].values)) > 1
                if self.dt_process[col_name+"_hour"]:
                    df['number_hour_{}'.format(col_name)] = hours
                    df['number_hour_of_week_{}'.format(col_name)] = hours + df[col_name].dt.weekday * 24
                #df['number_minute_of_day_{}'.format(col_name)] = df[col_name].apply(lambda x: x.minute + x.hour * 60)
        return df 

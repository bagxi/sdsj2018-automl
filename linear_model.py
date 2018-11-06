import time
import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from time_series_split import TimeSeriesCV


def linear_model(mode, **kwargs):
    if mode == 'classification':
        model = LinearClassification(**kwargs)
    else:
        model = LinearRegression(**kwargs)

    return model


def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


class LinearRegression:
    def __init__(self, random_seed=82):
        self.random_seed = random_seed
        self.transformer_params = {'random_state': self.random_seed + 1}
        self.transformer = QuantileTransformer(**self.transformer_params)
        self.model_params = {'max_iter': 1000, 'random_state': self.random_seed + 2}
        self.model = None

    def train(self, data, label, ds=None, train_tl=200):
        start_time = time.time()
        self.fillna_values = data.mean()
        data.fillna(self.fillna_values, inplace=True)
        self.model = Lasso(**self.model_params, alpha=0.1)
        self.model.fit(self.transformer.fit_transform(data), label)
        model_train_time = time.time() - start_time

        try:  # search
            if ds is not None:
                data['ds'] = ds
                cv = TimeSeriesCV(n_splits=min(6, data.shape[0] // 30))
                folds = list(cv.split(data))
                data.drop('ds', axis=1, inplace=True)
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=self.random_seed + 3)
                folds = list(cv.split(data))

            n_alphas = int(min(35, (train_tl - 2 * model_train_time) / (model_train_time * len(folds))))
            lasso_alpha, lasso_rmse = self._search_params(data, label, model=Lasso, search_space=np.logspace(-2, 0, n_alphas), folds=folds)
            Model, best_alpha = Lasso, lasso_alpha

            n_alphas = int(min(10, (train_tl - (time.time() - start_time) - model_train_time) / (model_train_time * 1.5 * len(folds))))
            if n_alphas > 2:
                ridge_alpha, ridge_rmse = self._search_params(data, label, model=Ridge, search_space=np.logspace(-2, 2, n_alphas), folds=folds)
                if lasso_rmse * 0.99 < ridge_rmse:
                    best_alpha = ridge_alpha
                    Model = Ridge

            self.model_params.update({'alpha': best_alpha, 'random_state': self.random_seed + 4})
            self.model = Model(**self.model_params)
            self.model.fit(self.transformer.transform(data), label)
        except:
            pass

    def predict(self, data):
        data = self.transformer.transform(data.fillna(self.fillna_values))
        preds = self.model.predict(data)

        return preds

    def _search_params(self, data, label, model, search_space, folds=3, scorer=None):
        scorer = scorer or make_scorer(_rmse, greater_is_better=False)
        pipeline = Pipeline([
            ('t', QuantileTransformer(**self.transformer_params)),
            ('m', model(**self.model_params))
        ])
        gs = GridSearchCV(pipeline, {'m__alpha': search_space}, scoring=scorer, cv=folds)
        gs.fit(data, label)

        return gs.best_params_['m__alpha'], gs.best_score_


class LinearClassification:
    def __init__(self, random_seed=82):
        self.random_seed = random_seed
        self.transformer_params = {'random_state': self.random_seed + 1}
        self.transformer = QuantileTransformer(**self.transformer_params)
        self.model_params = {
            'penalty': 'l2',
            'C': 5.0,
            'class_weight': 'balanced',
            'random_state': self.random_seed + 2,
            'solver': 'saga',
            'max_iter': 250,
            'n_jobs': 4,
        }
        self.model = None

    def train(self, data, label):
        self.fillna_values = data.mean()
        data = self.transformer.fit_transform(data.fillna(self.fillna_values))
        self.model = LogisticRegression(**self.model_params)
        self.model.fit(data, label)

    def predict(self, data):
        data = self.transformer.transform(data.fillna(self.fillna_values))
        preds = self.model.predict_proba(data)[:, 1]

        return preds

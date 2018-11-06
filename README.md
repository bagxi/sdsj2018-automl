# Sberbank Data Science Journey 2018: AutoML

[SDSJ AutoML](https://sdsj.sberbank.ai/ru/contest) — AutoML competition aimed at machine learning models that automatically process data, completely automatically choosing models, architectures, hyper-parameters, etc.

## Team members

[Dmitriy Kulagin](https://www.kaggle.com/rekcahd), [Yauheni Kachan](https://www.kaggle.com/xibagel), [Nastassia Smolskaya](https://www.kaggle.com/smolsnastya), [Vadim Yermakov](https://www.kaggle.com/zxspectrum)

## Solution description

### Preprocessing:

- Drop constant columns
- Add time-shifted columns
- Features from datetime columns (year, weekday, month, day)
- Smoothed target encoding (Semenov encoding) for `string` and `id` columns and if the dataset has more than 1000 rows else and for `numeric` with less than 31 unique values
- If dataset's size bigger than 250Mb than convert data to `np.float32` data-type

### Training:

If dataset has less than 1000 rows (e.g. first dataset) and `regression` problem than train [linear model](#Linear-Model) else [gradient boosting](#Gradient-Boosting) model(s)

#### Linear Model

1. Fill missing values with mean
2. Transform data with [QuantileTransformer][1]
3. Train [Lasso][2] with regularization term alpha = 0.1
4. Search for best alpha for Lasso, [Ridge][3] and select best of them:
  * Cross-validation: `time_series_split.TimeSeriesCV` with `min(6, number_of_rows / 30)` folds if `datetime_0` in dataset else use [KFold][4] with `3` folds
  * [Grid search][5] alpha for Lasso in range `np.logspace(-2, 0, n_points)` where `n_points` is min of 35 and estimation of how many times we could train the model on all folds
  * Grid search alpha Ridge if by estimation we could train more than 2 times on all folds, search for alpha in `np.logspace(-2, 2, n_points)` range
5. If we successfully grid search than select best of Lasso and Ridge else use Lasso from 2.

#### Gradient Boosting

- [XGBoost][6] with 700 trees (with `early_stopping_rounds=20`)
- If XGBoosts trains fewer two-thirds of available time than train [LightGBM][7] with 5000 trees (with `early_stopping_rounds=20`)
- If XGBoost and LightGBM trained successfully than stacking them with [Logistic Regression][8] or Ridge according to the prediction problem

[1]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
[2]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
[3]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
[4]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
[5]: http://scikit-learn.org/0.15/modules/generated/sklearn.grid_search.GridSearchCV.html
[6]: https://xgboost.readthedocs.io/en/latest/
[7]: https://github.com/Microsoft/LightGBM
[8]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

## Local Validation

- Official [how to](https://github.com/sberbank-ai/sdsj2018-automl/blob/master/README_EN.md#how-to-local-validation)
- Baseline [example](https://github.com/sberbank-ai/sdsj2018-automl)
- [@vlarine](https://github.com/vlarine)'s public [kernel](https://github.com/vlarine/sdsj2018_lightgbm_baseline).

Public datasets for local validation: [sdsj2018_automl_check_datasets.zip](https://s3.eu-central-1.amazonaws.com/sdsj2018-automl/public/sdsj2018_automl_check_datasets.zip)

### Docker :whale:

`docker pull rekcahd/sdsj2018`

## Useful links

- [SDSJ2018-AutoML](https://github.com/sberbank-ai/sdsj2018-automl)
- [LightGBM Baseline](https://github.com/vlarine/sdsj2018_lightgbm_baseline)
- [Sberbank Data Science Journey 2018: Docker-friendly baseline](https://github.com/tyz910/sdsj2018)
- [Leakage](https://github.com/bagxi/sdsj2018-leakage)

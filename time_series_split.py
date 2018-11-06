from datetime import timedelta


class TimeSeriesCV:
    def __init__(self, n_splits=5, n_test_days=30, ds_col='ds'):
        self.n_splits = n_splits
        self.test_size = n_test_days
        self.ds_col = ds_col

    def split(self, data, **kwargs):
        tdelta = timedelta(days=self.test_size)
        dmax = data[self.ds_col].max() - self.n_splits * tdelta
        for i in range(self.n_splits):
            train = data[(data[self.ds_col] < dmax)]
            test = data[(data[self.ds_col] >= dmax) & (data[self.ds_col] < dmax + tdelta)]
            train_idxs = train.index.values
            test_idxs = test.index.values
            yield train_idxs, test_idxs
            dmax += tdelta

    def get_n_splits(self, **kwargs):
        return self.n_splits

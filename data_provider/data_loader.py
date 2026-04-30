import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_provider.m4 import M4Dataset, M4Meta
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 train_only=None, seasonal_patterns=None):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data[border1s[0]:border2s[0]].values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month']   = df_stamp.date.apply(lambda r: r.month)
            df_stamp['day']     = df_stamp.date.apply(lambda r: r.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda r: r.weekday())
            df_stamp['hour']    = df_stamp.date.apply(lambda r: r.hour)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp['date'].values), freq=self.freq
            ).transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end   = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len
        return (self.data_x[s_begin:s_end],
                self.data_y[r_begin:r_end],
                self.data_stamp[s_begin:s_end],
                self.data_stamp[r_begin:r_end])

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 train_only=False, seasonal_patterns=None):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0,
                    12 * 30 * 24 * 4 - self.seq_len,
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4,
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                    12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data[border1s[0]:border2s[0]].values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month']   = df_stamp.date.apply(lambda r: r.month)
            df_stamp['day']     = df_stamp.date.apply(lambda r: r.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda r: r.weekday())
            df_stamp['hour']    = df_stamp.date.apply(lambda r: r.hour)
            df_stamp['minute']  = df_stamp.date.apply(lambda r: r.minute // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp['date'].values), freq=self.freq
            ).transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end   = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len
        return (self.data_x[s_begin:s_end],
                self.data_y[r_begin:r_end],
                self.data_stamp[s_begin:s_end],
                self.data_stamp[r_begin:r_end])

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 train_only=False, seasonal_patterns=None):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')

        num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
        num_test  = int(len(df_raw) * 0.2)
        num_vali  = len(df_raw) - num_train - num_test
        border1s  = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s  = [num_train, num_train + num_vali, len(df_raw)]
        border1   = border1s[self.set_type]
        border2   = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_raw  = df_raw[['date'] + cols]
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_raw  = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data[border1s[0]:border2s[0]].values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month']   = df_stamp.date.apply(lambda r: r.month)
            df_stamp['day']     = df_stamp.date.apply(lambda r: r.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda r: r.weekday())
            df_stamp['hour']    = df_stamp.date.apply(lambda r: r.hour)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp['date'].values), freq=self.freq
            ).transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end   = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len
        return (self.data_x[s_begin:s_end],
                self.data_y[r_begin:r_end],
                self.data_stamp[s_begin:s_end],
                self.data_stamp[r_begin:r_end])

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 train_only=False, seasonal_patterns=None):
        self.seq_len   = size[0]
        self.label_len = size[1]
        self.pred_len  = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale     = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), 'r', encoding='utf-8') as f:
            for line in f:
                data_line = np.array([float(x) for x in line.strip('\n').split(',')])
                df_raw.append(data_line)
        df_raw = pd.DataFrame(np.stack(df_raw, 0))

        num_train = int(len(df_raw) * 0.7)
        num_test  = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s  = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s  = [num_train, num_train + num_valid, len(df_raw)]
        b1, b2    = border1s[self.set_type], border2s[self.set_type]

        data = df_raw.values
        if self.scale:
            self.scaler.fit(data[border1s[0]:border2s[0]])
            data = self.scaler.transform(data)

        self.data_x = data[b1:b2]
        self.data_y = data[b1:b2]

    def __getitem__(self, index):
        s_begin = index
        s_end   = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len
        seq_x   = self.data_x[s_begin:s_end]
        seq_y   = self.data_y[r_begin:r_end]
        return (seq_x, seq_y,
                torch.zeros((seq_x.shape[0], 1)),
                torch.zeros((seq_y.shape[0], 1)))

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0,
                 freq='15min', cols=None, train_only=False, seasonal_patterns=None):
        if size is None:
            self.seq_len   = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len  = 24 * 4
        else:
            self.seq_len   = size[0]
            self.label_len = size[1]
            self.pred_len  = size[2]
        assert flag in ['pred']
        self.features  = features
        self.target    = target
        self.scale     = scale
        self.inverse   = inverse
        self.timeenc   = timeenc
        self.freq      = freq
        self.cols      = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
            cols.remove('date')
        if self.features == 'S':
            cols.remove(self.target)

        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            df_raw  = df_raw[['date'] + cols]
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_raw  = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp  = df_raw[['date']][border1:border2].copy()
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(
            tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame({'date': list(tmp_stamp.date.values) + list(pred_dates[1:])})
        self.future_dates = list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month']   = df_stamp.date.apply(lambda r: r.month)
            df_stamp['day']     = df_stamp.date.apply(lambda r: r.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda r: r.weekday())
            df_stamp['hour']    = df_stamp.date.apply(lambda r: r.hour)
            df_stamp['minute']  = df_stamp.date.apply(lambda r: r.minute // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp['date'].values), freq=self.freq
            ).transpose(1, 0)

        self.data_x    = data[border1:border2]
        self.data_y    = df_data.values[border1:border2] if self.inverse else data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end   = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len
        seq_y   = (self.data_x if self.inverse else self.data_y)[r_begin:r_begin + self.label_len]
        return (self.data_x[s_begin:s_end], seq_y,
                self.data_stamp[s_begin:s_end],
                self.data_stamp[r_begin:r_end])

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0,
                 freq='15min', seasonal_patterns='Yearly', train_only=False):
        self.seq_len   = size[0]
        self.label_len = size[1]
        self.pred_len  = size[2]
        self.seasonal_patterns     = seasonal_patterns
        self.history_size          = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.root_path = root_path
        self.flag      = flag
        self.__read_data__()

    def __read_data__(self):
        dataset = M4Dataset.load(
            training=(self.flag == 'train'), dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]],
            dtype=object)
        self.ids        = dataset.ids[dataset.groups == self.seasonal_patterns]
        self.timeseries = list(training_values)

    def __getitem__(self, index):
        insample       = np.zeros((self.seq_len, 1))
        insample_mask  = np.zeros((self.seq_len, 1))
        outsample      = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))

        ts = self.timeseries[index]
        cut = np.random.randint(
            low=max(1, len(ts) - self.window_sampling_limit),
            high=len(ts), size=1)[0]

        in_win = ts[max(0, cut - self.seq_len):cut]
        insample[-len(in_win):, 0]      = in_win
        insample_mask[-len(in_win):, 0] = 1.0

        out_win = ts[cut - self.label_len:min(len(ts), cut + self.pred_len)]
        outsample[:len(out_win), 0]      = out_win
        outsample_mask[:len(out_win), 0] = 1.0

        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def last_insample_window(self):
        """Returns (n_series, seq_len) arrays for test-time prediction."""
        insample      = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            win = ts[-self.seq_len:]
            insample[i, -len(win):]      = win
            insample_mask[i, -len(win):] = 1.0
        return insample, insample_mask


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, train_only=False):
        self.seq_len   = size[0]
        self.label_len = size[1]
        self.pred_len  = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type  = type_map[flag]
        self.scale     = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        raw = np.load(os.path.join(self.root_path, self.data_path),
                      allow_pickle=True)['data'][:, :, 0]

        train_r, valid_r = 0.6, 0.2
        n = len(raw)
        borders = [
            (0,                          int(train_r * n)),
            (int(train_r * n),           int((train_r + valid_r) * n)),
            (int((train_r + valid_r)*n), n),
        ]
        b1, b2 = borders[self.set_type]

        if self.scale:
            self.scaler.fit(raw[borders[0][0]:borders[0][1]])
            raw = self.scaler.transform(raw)

        data = pd.DataFrame(raw).ffill().bfill().values
        self.data_x = data[b1:b2]
        self.data_y = data[b1:b2]

    def __getitem__(self, index):
        # test: non-overlapping windows every 12 steps
        s = index * 12 if self.set_type == 2 else index
        seq_x = self.data_x[s:s + self.seq_len]
        seq_y = self.data_y[s + self.seq_len - self.label_len:
                            s + self.seq_len + self.pred_len]
        return (seq_x, seq_y,
                torch.zeros((seq_x.shape[0], 1)),
                torch.zeros((seq_y.shape[0], 1)))

    def __len__(self):
        if self.set_type == 2:
            return (len(self.data_x) - self.seq_len - self.pred_len + 1) // 12
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

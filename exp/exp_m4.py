"""
exp/exp_m4.py — M4 Short-Term Forecasting Experiment for PatchLinear
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_M4
from data_provider.m4 import M4Meta
from utils.losses import smape_loss
from utils.m4_summary import M4Summary


def _get_model(args):
    from models.PatchLinear import Model
    args.enc_in = 1
    args.use_cross_channel = False
    args.use_alpha_gate = False
    return Model(args)


def _adjust_lr(optimizer, epoch, args):
    """type3 step decay: lr = lr0 * 0.9^epoch"""
    lr = args.learning_rate * (0.9 ** epoch)
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def _count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Exp_M4:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            f'cuda:{args.gpu}' if args.use_gpu and torch.cuda.is_available() else 'cpu'
        )
        self.model = _get_model(args).to(self.device)
        self.loss_fn = smape_loss()

    def _loader(self, flag):
        dataset = Dataset_M4(
            root_path=self.args.root_path,
            flag=flag,
            size=[self.args.seq_len, 0, self.args.pred_len],
            seasonal_patterns=self.args.seasonal_patterns,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=(flag == 'train'),
            num_workers=self.args.num_workers,
            drop_last=False,
        )
        return dataset, loader

    def train(self, setting):
        train_dataset, train_loader = self._loader('train')

        # ── print config (matches style of exp_main.py) ───────────────────────
        n_series = len(train_dataset)
        n_params  = _count_params(self.model)
        print(f'  frequency      : {self.args.seasonal_patterns}')
        print(f'  seq_len        : {self.args.seq_len}')
        print(f'  pred_len       : {self.args.pred_len}')
        print(f'  train series   : {n_series}')
        print(f'  d_model        : {self.args.d_model}')
        print(f'  patch/stride/k : {self.args.patch_len}/{self.args.stride}/{self.args.dw_kernel}')
        print(f'  decomp/seas/trend/fusion: '
              f'{int(self.args.use_decomp)}/'
              f'{int(self.args.use_seas_stream)}/'
              f'{int(self.args.use_trend_stream)}/'
              f'{int(self.args.use_fusion_gate)}')
        print(f'  parameters     : {n_params:,}')
        print(f'  device         : {self.device}')
        print(f'  epochs/lr/bs   : {self.args.train_epochs} / '
              f'{self.args.learning_rate} / {self.args.batch_size}')
        print()

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        freq      = M4Meta.frequency_map[self.args.seasonal_patterns]

        best_loss  = float('inf')
        os.makedirs(self.args.checkpoints, exist_ok=True)
        ckpt_path  = os.path.join(self.args.checkpoints, f'{setting}.pth')

        for epoch in range(self.args.train_epochs):
            lr = _adjust_lr(optimizer, epoch, self.args)
            self.model.train()
            epoch_loss = []
            t_start    = time.time()

            for insample, outsample, insample_mask, outsample_mask in train_loader:
                insample       = insample.float().to(self.device)
                outsample      = outsample.float().to(self.device)
                insample_mask  = insample_mask.float().to(self.device)
                outsample_mask = outsample_mask.float().to(self.device)

                optimizer.zero_grad()
                masked_input = insample * insample_mask
                forecast     = self.model(masked_input)

                loss = self.loss_fn(
                    insample.squeeze(-1),
                    freq,
                    forecast.squeeze(-1),
                    outsample.squeeze(-1),
                    outsample_mask.squeeze(-1),
                )
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())

            mean_loss  = np.mean(epoch_loss)
            elapsed    = time.time() - t_start
            is_best    = mean_loss < best_loss

            if is_best:
                best_loss = mean_loss
                torch.save(self.model.state_dict(), ckpt_path)

            # ── per-epoch log (every epoch, like exp_main.py) ────────────────
            print(f'  Epoch {epoch+1:>3}/{self.args.train_epochs} | '
                  f'lr: {lr:.2e} | '
                  f'train smape: {mean_loss:.4f} | '
                  f'time: {elapsed:.1f}s'
                  + (' ★' if is_best else ''))

        print(f'\n  Best train smape: {best_loss:.4f}')
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        return self.model

    def predict(self, setting, forecast_dir):
        train_dataset = Dataset_M4(
            root_path=self.args.root_path,
            flag='train',
            size=[self.args.seq_len, 0, self.args.pred_len],
            seasonal_patterns=self.args.seasonal_patterns,
        )
        self.model.eval()
        os.makedirs(forecast_dir, exist_ok=True)

        insample_np, insample_mask_np = train_dataset.last_insample_window()
        n_series    = insample_np.shape[0]
        batch_size  = 256
        all_forecasts = []

        print(f'\n  Predicting {n_series} test series...')
        with torch.no_grad():
            for start in range(0, n_series, batch_size):
                end = min(start + batch_size, n_series)
                x = torch.tensor(
                    insample_np[start:end, :, np.newaxis],
                    dtype=torch.float32, device=self.device)
                m = torch.tensor(
                    insample_mask_np[start:end, :, np.newaxis],
                    dtype=torch.float32, device=self.device)
                forecast = self.model(x * m)
                all_forecasts.append(forecast.squeeze(-1).cpu().numpy())

        all_forecasts = np.concatenate(all_forecasts, axis=0)
        csv_path = os.path.join(forecast_dir, f'{self.args.seasonal_patterns}_forecast.csv')
        pd.DataFrame(all_forecasts).to_csv(csv_path, index=False)
        print(f'  Saved → {csv_path}')

    def evaluate(self, forecast_dir):
        summary = M4Summary(
            file_path=os.path.join(forecast_dir, ''),
            root_path=self.args.root_path,
        )
        smapes, owas, mapes, mases = summary.evaluate()
        return smapes, owas

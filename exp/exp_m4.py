"""
exp/exp_m4.py — M4 Short-Term Forecasting Experiment for PatchLinear

Place this file at: exp/exp_m4.py

Usage: called by run_m4.py, not by the standard run.py.

Protocol:
  - Train with smape_loss on Dataset_M4 (masked, zero-padded series)
  - Predict on last_insample_window() for all test series
  - Save per-frequency forecast CSVs for M4Summary.evaluate()
  - Report SMAPE and OWA per frequency
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_M4
from data_provider.m4 import M4Meta
from utils.losses import smape_loss
from utils.m4_summary import M4Summary


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_model(args):
    """Instantiate PatchLinear with M4-appropriate config."""
    from models.PatchLinear import Model
    # Override for M4: univariate, no cross-channel, short seasonal patches
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


# ── experiment class ──────────────────────────────────────────────────────────

class Exp_M4:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            f'cuda:{args.gpu}' if args.use_gpu and torch.cuda.is_available() else 'cpu'
        )
        self.model = _get_model(args).to(self.device)
        self.loss_fn = smape_loss()

    # ── data ─────────────────────────────────────────────────────────────────

    def _loader(self, flag):
        dataset = Dataset_M4(
            root_path=self.args.root_path,
            flag=flag,
            size=[self.args.seq_len, 0, self.args.pred_len],
            seasonal_patterns=self.args.seasonal_patterns,
        )
        shuffle = (flag == 'train')
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            drop_last=False,
        )
        return dataset, loader

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self, setting):
        _, train_loader = self._loader('train')

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        freq = M4Meta.frequency_map[self.args.seasonal_patterns]

        best_loss = float('inf')
        os.makedirs(self.args.checkpoints, exist_ok=True)
        ckpt_path = os.path.join(self.args.checkpoints, f'{setting}.pth')

        for epoch in range(self.args.train_epochs):
            lr = _adjust_lr(optimizer, epoch, self.args)
            self.model.train()
            epoch_loss = []

            for insample, outsample, insample_mask, outsample_mask in train_loader:
                # insample:       (B, seq_len, 1)  — zero-padded
                # insample_mask:  (B, seq_len, 1)  — 1 where real data
                # outsample:      (B, pred_len, 1)
                # outsample_mask: (B, pred_len, 1)
                insample      = insample.float().to(self.device)
                outsample     = outsample.float().to(self.device)
                insample_mask = insample_mask.float().to(self.device)
                outsample_mask= outsample_mask.float().to(self.device)

                optimizer.zero_grad()

                # Apply input mask (zero out padded region, model sees masked input)
                masked_input = insample * insample_mask  # (B, seq_len, 1)

                # Forward pass — PatchLinear expects (B, L, C)
                forecast = self.model(masked_input)      # (B, pred_len, 1)

                # smape_loss signature: (insample, freq, forecast, target, mask)
                # Squeeze channel dim for loss computation
                loss = self.loss_fn(
                    insample.squeeze(-1),        # (B, seq_len)
                    freq,
                    forecast.squeeze(-1),        # (B, pred_len)
                    outsample.squeeze(-1),       # (B, pred_len)
                    outsample_mask.squeeze(-1),  # (B, pred_len)
                )

                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())

            mean_loss = np.mean(epoch_loss)
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(self.model.state_dict(), ckpt_path)

            if (epoch + 1) % 10 == 0:
                print(f'  Epoch {epoch+1}/{self.args.train_epochs}  '
                      f'lr={lr:.2e}  smape_loss={mean_loss:.4f}')

        # Restore best checkpoint
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        return self.model

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, setting, forecast_dir):
        """
        Generate forecasts for all test series and save per-frequency CSV.
        M4Summary.evaluate() expects: {forecast_dir}/{Frequency}_forecast.csv
        Each row = one series, columns = forecast steps.
        """
        # Load all training series for this frequency to get last window
        train_dataset = Dataset_M4(
            root_path=self.args.root_path,
            flag='train',
            size=[self.args.seq_len, 0, self.args.pred_len],
            seasonal_patterns=self.args.seasonal_patterns,
        )

        self.model.eval()
        os.makedirs(forecast_dir, exist_ok=True)

        # last_insample_window: (n_series, seq_len) with zero-padding
        insample_np, insample_mask_np = train_dataset.last_insample_window()
        # shape: (n_series, seq_len)

        n_series = insample_np.shape[0]
        batch_size = 256  # process in chunks to fit GPU
        all_forecasts = []

        with torch.no_grad():
            for start in range(0, n_series, batch_size):
                end = min(start + batch_size, n_series)
                x = torch.tensor(
                    insample_np[start:end, :, np.newaxis],  # (B, seq_len, 1)
                    dtype=torch.float32, device=self.device
                )
                m = torch.tensor(
                    insample_mask_np[start:end, :, np.newaxis],
                    dtype=torch.float32, device=self.device
                )
                masked_x = x * m
                forecast = self.model(masked_x)  # (B, pred_len, 1)
                all_forecasts.append(forecast.squeeze(-1).cpu().numpy())

        all_forecasts = np.concatenate(all_forecasts, axis=0)  # (n_series, pred_len)

        # Save as CSV — M4Summary expects this format
        freq_name = self.args.seasonal_patterns
        import pandas as pd
        df = pd.DataFrame(all_forecasts)
        csv_path = os.path.join(forecast_dir, f'{freq_name}_forecast.csv')
        df.to_csv(csv_path, index=False)
        print(f'  Saved {n_series} forecasts → {csv_path}')

    # ── evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self, forecast_dir):
        """
        Run M4Summary evaluation and print SMAPE / OWA per frequency.
        Requires submission-Naive2.csv in root_path.
        """
        summary = M4Summary(
            file_path=os.path.join(forecast_dir, ''),
            root_path=self.args.root_path,
        )
        smapes, owas, mapes, mases = summary.evaluate()
        return smapes, owas

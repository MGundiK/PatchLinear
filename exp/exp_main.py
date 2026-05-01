from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import PatchLinear
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import math

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        model_dict = {
            'PatchLinear': PatchLinear,
        }
        if self.args.model not in model_dict:
            raise ValueError(
                f"Unknown model '{self.args.model}'. "
                f"Available: {list(model_dict.keys())}"
            )
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        huber_delta = getattr(self.args, 'huber_delta', 1.0)
        huber = nn.SmoothL1Loss(beta=huber_delta)
        return nn.MSELoss(), nn.L1Loss(), huber

    def _arctan_ratio(self, pred_len, device):
        ratio = np.array(
            [-math.atan(i + 1) + math.pi / 4 + 1 for i in range(pred_len)]
        )
        return torch.tensor(ratio).unsqueeze(-1).to(device)

    def _compute_train_loss(self, outputs, batch_y, ratio,
                            mse_criterion, mae_criterion, huber_criterion):
        kind = getattr(self.args, 'train_loss', 'wmae')
        if kind == 'wmae':
            return mae_criterion(outputs * ratio, batch_y * ratio)
        if kind == 'mae':
            return mae_criterion(outputs, batch_y)
        if kind == 'mse':
            return mse_criterion(outputs, batch_y)
        if kind == 'huber':
            return huber_criterion(outputs, batch_y)
        raise ValueError(f"Unknown train_loss: {kind}")

    def vali(self, vali_loader, criterion, use_loss_weight=False):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)
                f_dim   = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                if use_loss_weight:
                    ratio   = self._arctan_ratio(self.args.pred_len, self.device)
                    outputs = outputs * ratio
                    batch_y = batch_y * ratio
                total_loss.append(criterion(outputs, batch_y).item())
        self.model.train()
        return np.mean(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        vali_data,  vali_loader  = self._get_data('val')
        test_data,  test_loader  = self._get_data('test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps    = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim    = self._select_optimizer()
        mse_criterion, mae_criterion, huber_criterion = self._select_criterion()

        train_loss_kind   = getattr(self.args, 'train_loss',        'wmae')
        early_stop_metric = getattr(self.args, 'early_stop_metric', 'wmae')
        print(
            f"[loss] train={train_loss_kind}  "
            f"early_stop_metric={early_stop_metric}  "
            f"huber_delta={getattr(self.args, 'huber_delta', 1.0)}"
        )

        for epoch in range(self.args.train_epochs):
            train_loss  = []
            iter_count  = 0
            time_now    = time.time()
            epoch_start = time.time()
            self.model.train()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)
                f_dim   = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                ratio   = self._arctan_ratio(self.args.pred_len, self.device)
                loss    = self._compute_train_loss(
                    outputs, batch_y, ratio,
                    mse_criterion, mae_criterion, huber_criterion,
                )
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

                if (i + 1) % 100 == 0:
                    speed     = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(
                        f"\titers: {i+1}, epoch: {epoch+1} | "
                        f"loss: {loss.item():.6f} | "
                        f"speed: {speed:.4f}s/iter | left: {left_time:.1f}s"
                    )
                    iter_count = 0
                    time_now   = time.time()

            train_loss = np.mean(train_loss)

            vali_wmae = self.vali(vali_loader, mae_criterion, use_loss_weight=True)
            vali_mae  = self.vali(vali_loader, mae_criterion, use_loss_weight=False)
            vali_mse  = self.vali(vali_loader, mse_criterion, use_loss_weight=False)
            test_mse  = self.vali(test_loader,  mse_criterion, use_loss_weight=False)

            es_signal = {
                'wmae': vali_wmae, 'mae': vali_mae, 'mse': vali_mse,
            }[early_stop_metric]

            print(
                f"Epoch {epoch+1} | "
                f"time: {time.time()-epoch_start:.1f}s | "
                f"train({train_loss_kind}): {train_loss:.6f} | "
                f"vali(wMAE): {vali_wmae:.6f} | vali(MAE): {vali_mae:.6f} | "
                f"vali(MSE): {vali_mse:.6f} | test(MSE): {test_mse:.6f} | "
                f"ES({early_stop_metric}): {es_signal:.6f}"
            )

            early_stopping(es_signal, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        if not getattr(self.args, 'keep_checkpoint', False):
            os.remove(best_model_path)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data('test')
        if test:
            ckpt = os.path.join('./checkpoints', setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(ckpt))

        folder_path = os.path.join('./test_results', setting)
        os.makedirs(folder_path, exist_ok=True)

        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)
                f_dim   = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()
                preds.append(outputs)
                trues.append(batch_y)
                if i % 20 == 0:
                    inp = batch_x.detach().cpu().numpy()
                    gt  = np.concatenate((inp[0, :, -1], batch_y[0, :, -1]))
                    pd  = np.concatenate((inp[0, :, -1], outputs[0, :, -1]))
                    visual(gt, pd, os.path.join(folder_path, f'{i}.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # ── PEMS: inverse-transform to raw traffic scale before metrics ───────
        # Matches the TimeMixer / TimeMixer++ evaluation protocol exactly:
        # train/val in standardized space, test reported in raw scale.
        # No effect on any other dataset — the branch is PEMS-only.
        if self.args.data == 'PEMS':
            B, T, C = preds.shape
            preds = test_data.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
            trues = test_data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print(f'mse:{mse:.6f}, mae:{mae:.6f}, rmse:{rmse:.6f}, mape:{mape:.6f}')
            with open('result.txt', 'a') as f:
                f.write(f'{setting}\n')
                f.write(f'mse:{mse:.6f}, mae:{mae:.6f}, rmse:{rmse:.6f}, mape:{mape:.6f}\n\n')
        else:
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print(f'mse:{mse:.6f}, mae:{mae:.6f}')
            with open('result.txt', 'a') as f:
                f.write(f'{setting}\n')
                f.write(f'mse:{mse:.6f}, mae:{mae:.6f}\n\n')

    def analyse_alpha(self, n_batches=10):
        """
        Compute mean alpha per channel over n_batches of test data.
        Used for the interpretability figure (A4b / A5).
        Returns tensor [C] of mean alpha values.
        """
        assert self.args.use_cross_channel and self.args.use_alpha_gate, \
            "Alpha gate must be active (use_cross_channel=1 and use_alpha_gate=1)."
        _, test_loader = self._get_data('test')
        all_alphas = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, *_) in enumerate(test_loader):
                if i >= n_batches:
                    break
                batch_x = batch_x.float().to(self.device)
                x = self.model.revin(batch_x, 'norm')
                if self.model.use_decomp:
                    seasonal, trend = self.model.decomp(x)
                else:
                    seasonal = trend = x
                _, alpha = self.model.backbone.get_alpha_values(
                    seasonal.permute(0, 2, 1),
                    trend.permute(0, 2, 1),
                )
                all_alphas.append(alpha.squeeze(-1).mean(dim=0).cpu())
        return torch.stack(all_alphas).mean(dim=0)

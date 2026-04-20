from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import xPatch
from models import PatchLinear
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import autocast, GradScaler
import os
import csv
import time
import warnings
import math

warnings.filterwarnings('ignore')




# Model registry: add new models here without touching the rest of the file.



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
        return nn.MSELoss(), nn.L1Loss()

    def _arctan_ratio(self, pred_len, device):
        # Arctangent loss weight schedule from xPatch.
        # rho(i) = -arctan(i+1) + pi/4 + 1
        # Downweights far-future steps where variance is highest.
        ratio = np.array(
            [-math.atan(i + 1) + math.pi / 4 + 1 for i in range(pred_len)]
        )
        return torch.tensor(ratio).unsqueeze(-1).to(device)

    # ──────────────────────────────────────────────────────────────────────
    # Helpers added for SWA / cosine warm restarts / drift logging
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _spearman(a, b):
        """Rank correlation between two equally-sized 1D sequences.
        Returns NaN if fewer than 2 valid (non-NaN) pairs or zero variance.
        """
        n = min(len(a), len(b))
        if n < 2:
            return float('nan')
        a = np.asarray(a[-n:], dtype=float)
        b = np.asarray(b[-n:], dtype=float)
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 2:
            return float('nan')
        a, b = a[mask], b[mask]
        ra = a.argsort().argsort()
        rb = b.argsort().argsort()
        if np.std(ra) == 0 or np.std(rb) == 0:
            return float('nan')
        return float(np.corrcoef(ra, rb)[0, 1])

    @staticmethod
    def _has_batchnorm(module):
        bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
        return any(isinstance(m, bn_types) for m in module.modules())

    def _update_bn(self, loader, model):
        """Drop-in replacement for torch.optim.swa_utils.update_bn that
        respects our loader's tuple format and float casting.

        The stock update_bn passes batch[0] straight to the model without
        calling .float(), which breaks when the dataloader emits float64.
        This version matches the dtype/device handling of our train loop.
        """
        bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
        momenta = {}
        for m in model.modules():
            if isinstance(m, bn_types):
                m.reset_running_stats()
                momenta[m] = m.momentum
        if not momenta:
            return  # no BN layers; nothing to update

        was_training = model.training
        model.train()
        n = 0
        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
                batch_x = batch_x.float().to(self.device)
                b = batch_x.size(0)
                # Running-mean momentum weighted by batch size across the loader.
                momentum = b / float(n + b)
                for m in momenta:
                    m.momentum = momentum
                model(batch_x)
                n += b

        # Restore the original momentum values
        for m, mom in momenta.items():
            m.momentum = mom
        model.train(was_training)

    def _save_history(self, history, path):
        csv_path = os.path.join(path, 'history.csv')
        keys = list(history.keys())
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(keys)
            for i in range(len(history['epoch'])):
                w.writerow([history[k][i] for k in keys])
        print(f"[history] wrote {csv_path}")

    # ──────────────────────────────────────────────────────────────────────

    def vali(self, vali_loader, criterion, use_loss_weight=False, model=None):
        """Evaluate a model on a loader. If `model` is None, uses self.model.
        Passing an external model (e.g. AveragedModel) does not touch
        self.model's train/eval state.
        """
        model = self.model if model is None else model
        is_self = (model is self.model)

        total_loss = []
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = model(batch_x)
                f_dim   = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                if use_loss_weight:
                    ratio   = self._arctan_ratio(self.args.pred_len, self.device)
                    outputs = outputs * ratio
                    batch_y = batch_y * ratio
                total_loss.append(criterion(outputs, batch_y).item())
        if is_self:
            model.train()
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
        mse_criterion, mae_criterion = self._select_criterion()

        # ── scheduler setup ────────────────────────────────────────────────
        use_cosine = (self.args.lradj == 'cosine_warm')
        cosine_scheduler = None
        if use_cosine:
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                model_optim,
                T_0=self.args.cosine_T_0,
                T_mult=self.args.cosine_T_mult,
                eta_min=self.args.cosine_eta_min,
            )
            print(
                f"[scheduler] CosineAnnealingWarmRestarts "
                f"T_0={self.args.cosine_T_0}, T_mult={self.args.cosine_T_mult}, "
                f"eta_min={self.args.cosine_eta_min}"
            )

        # ── SWA setup ──────────────────────────────────────────────────────
        use_swa = getattr(self.args, 'use_swa', False)
        swa_start_epoch = int(
            self.args.train_epochs * getattr(self.args, 'swa_start_frac', 0.75)
        )
        swa_lr = getattr(self.args, 'swa_lr', None)
        if swa_lr is None:
            swa_lr = 0.05 * self.args.learning_rate
        swa_anneal_epochs = getattr(self.args, 'swa_anneal_epochs', 1)
        swa_model       = None
        swa_scheduler   = None
        swa_active      = False
        if use_swa:
            print(
                f"[SWA] enabled: starts at epoch {swa_start_epoch + 1}/"
                f"{self.args.train_epochs}, swa_lr={swa_lr:.2e}, "
                f"anneal_epochs={swa_anneal_epochs}"
            )

        # ── history for drift logging ──────────────────────────────────────
        history = {
            'epoch':         [],
            'lr':            [],
            'train':         [],
            'vali_wmae':     [],
            'vali_mse':      [],
            'test_mse':      [],
            'swa_vali_wmae': [],
            'swa_vali_mse':  [],
            'swa_test_mse':  [],
        }

        for epoch in range(self.args.train_epochs):
            # ── SWA activation (at the start of the designated epoch) ─────
            if use_swa and (not swa_active) and epoch >= swa_start_epoch:
                swa_active = True
                swa_model  = torch.optim.swa_utils.AveragedModel(self.model)
                swa_scheduler = torch.optim.swa_utils.SWALR(
                    model_optim,
                    swa_lr=swa_lr,
                    anneal_epochs=swa_anneal_epochs,
                    anneal_strategy='linear',
                )
                print(
                    f"[SWA] activated at epoch {epoch + 1}; "
                    f"early stopping suspended; LR ramping to swa_lr."
                )

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
                loss    = mae_criterion(outputs * ratio, batch_y * ratio)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

                # Per-batch cosine scheduler step (pre-SWA phase only).
                # Fractional epoch gives a smooth curve across batches.
                if use_cosine and not swa_active:
                    cosine_scheduler.step(epoch + i / train_steps)

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

            # ── end of epoch: SWA accumulation ────────────────────────────
            if swa_active:
                swa_model.update_parameters(self.model)

            train_loss = np.mean(train_loss)

            # ── validation & test (regular model) ─────────────────────────
            # vali(wMAE) matches training objective and is the early-stop signal.
            # vali(MSE) and test(MSE) are added so drift between signals is visible.
            vali_wmae = self.vali(vali_loader, mae_criterion, use_loss_weight=True)
            vali_mse  = self.vali(vali_loader, mse_criterion, use_loss_weight=False)
            test_mse  = self.vali(test_loader, mse_criterion, use_loss_weight=False)

            # ── validation & test (SWA model) ─────────────────────────────
            swa_vali_wmae = float('nan')
            swa_vali_mse  = float('nan')
            swa_test_mse  = float('nan')
            if swa_active:
                swa_vali_wmae = self.vali(vali_loader, mae_criterion,
                                          use_loss_weight=True,  model=swa_model)
                swa_vali_mse  = self.vali(vali_loader, mse_criterion,
                                          use_loss_weight=False, model=swa_model)
                swa_test_mse  = self.vali(test_loader, mse_criterion,
                                          use_loss_weight=False, model=swa_model)

            # ── record history ────────────────────────────────────────────
            history['epoch'].append(epoch + 1)
            history['lr'].append(model_optim.param_groups[0]['lr'])
            history['train'].append(float(train_loss))
            history['vali_wmae'].append(float(vali_wmae))
            history['vali_mse'].append(float(vali_mse))
            history['test_mse'].append(float(test_mse))
            history['swa_vali_wmae'].append(float(swa_vali_wmae))
            history['swa_vali_mse'].append(float(swa_vali_mse))
            history['swa_test_mse'].append(float(swa_test_mse))

            # ── drift diagnostic ──────────────────────────────────────────
            # Spearman rank corr between vali(wMAE) and test(MSE). Both are
            # lowest-is-best, so +1 = signals agree, ~0 = uncorrelated,
            # negative = early-stop signal points opposite to benchmark.
            drift_all    = self._spearman(history['vali_wmae'], history['test_mse'])
            drift_recent = self._spearman(
                history['vali_wmae'][-5:], history['test_mse'][-5:]
            )

            msg = (
                f"Epoch {epoch+1} | "
                f"time: {time.time()-epoch_start:.1f}s | "
                f"lr: {history['lr'][-1]:.2e} | "
                f"train: {train_loss:.6f} | "
                f"vali(wMAE): {vali_wmae:.6f} | vali(MSE): {vali_mse:.6f} | "
                f"test(MSE): {test_mse:.6f} | "
                f"drift(all/last5): {drift_all:+.2f}/{drift_recent:+.2f}"
            )
            if swa_active:
                msg += (
                    f"\n         [SWA] vali(wMAE): {swa_vali_wmae:.6f} | "
                    f"vali(MSE): {swa_vali_mse:.6f} | "
                    f"test(MSE): {swa_test_mse:.6f}"
                )
            print(msg)

            if len(history['epoch']) >= 5 and not np.isnan(drift_recent) and drift_recent < -0.3:
                print(
                    f"  ⚠  drift(last5)={drift_recent:+.2f}: vali(wMAE) is "
                    f"moving OPPOSITE to test(MSE) recently. Early-stopping "
                    f"signal may be misleading on this run."
                )

            # ── early stopping (suspended during SWA phase) ───────────────
            if not swa_active:
                early_stopping(vali_wmae, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            # ── LR step at end of epoch ───────────────────────────────────
            if swa_active:
                swa_scheduler.step()
            elif not use_cosine:
                # Legacy dict-based schedules (sigmoid, type1/2/3, etc.)
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            # else: cosine already stepped per-batch above

        # ── dump history for offline inspection ────────────────────────────
        self._save_history(history, path)

        # ── final model selection ──────────────────────────────────────────
        best_ckpt = os.path.join(path, 'checkpoint.pth')
        swa_ckpt  = os.path.join(path, 'swa_checkpoint.pth')

        if use_swa and swa_active and swa_model is not None:
            # Refresh BatchNorm running stats on the averaged weights.
            # For models without BN this is a no-op; skip to save time.
            if self._has_batchnorm(swa_model):
                print("[SWA] updating BN running statistics...")
                self._update_bn(train_loader, swa_model)
            else:
                print("[SWA] no BatchNorm layers detected; skipping update_bn.")

            # Persist SWA weights
            torch.save(swa_model.module.state_dict(), swa_ckpt)

            # Evaluate regular-best checkpoint
            self.model.load_state_dict(torch.load(best_ckpt))
            reg_vali_wmae = self.vali(vali_loader, mae_criterion, use_loss_weight=True)
            reg_vali_mse  = self.vali(vali_loader, mse_criterion, use_loss_weight=False)
            reg_test_mse  = self.vali(test_loader, mse_criterion, use_loss_weight=False)

            # Evaluate SWA weights
            self.model.load_state_dict(torch.load(swa_ckpt))
            fin_swa_vali_wmae = self.vali(vali_loader, mae_criterion, use_loss_weight=True)
            fin_swa_vali_mse  = self.vali(vali_loader, mse_criterion, use_loss_weight=False)
            fin_swa_test_mse  = self.vali(test_loader, mse_criterion, use_loss_weight=False)

            print("─" * 72)
            print(
                f"[Final] Regular  vali(wMAE) {reg_vali_wmae:.6f}  "
                f"vali(MSE) {reg_vali_mse:.6f}  test(MSE) {reg_test_mse:.6f}"
            )
            print(
                f"[Final] SWA      vali(wMAE) {fin_swa_vali_wmae:.6f}  "
                f"vali(MSE) {fin_swa_vali_mse:.6f}  test(MSE) {fin_swa_test_mse:.6f}"
            )

            # Select by vali(wMAE) — stays consistent with early-stopping signal
            # and avoids selecting on the test set.
            if fin_swa_vali_wmae < reg_vali_wmae:
                print("[Final] SWA wins on vali(wMAE) — keeping SWA weights.")
                torch.save(swa_model.module.state_dict(), best_ckpt)
                # self.model already has SWA weights
            else:
                print("[Final] Regular wins on vali(wMAE) — reverting to regular best.")
                self.model.load_state_dict(torch.load(best_ckpt))

            if not getattr(self.args, 'keep_checkpoint', False):
                if os.path.exists(swa_ckpt):
                    os.remove(swa_ckpt)
        else:
            # No SWA: keep existing behavior exactly
            self.model.load_state_dict(torch.load(best_ckpt))

        if not getattr(self.args, 'keep_checkpoint', False):
            if os.path.exists(best_ckpt):
                os.remove(best_ckpt)

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
        mae, mse = metric(preds, trues)
        print(f'mse: {mse:.6f}  mae: {mae:.6f}')
        with open('result.txt', 'a') as f:
            f.write(f'{setting}\n')
            f.write(f'mse:{mse:.6f}, mae:{mae:.6f}\n\n')

    def analyse_alpha(self, n_batches=10):
        """
        Compute mean alpha per channel over n_batches of test data.
        Used for the interpretability figure (A4b / A5).

        Returns tensor [C] of mean alpha values.
        Expected: high values on Traffic, low values on Exchange.
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
                # Reproduce the forward pre-processing to reach get_alpha_values
                x = self.model.revin(batch_x, 'norm')
                if self.model.use_decomp:
                    seasonal, trend = self.model.decomp(x)
                else:
                    seasonal = trend = x
                _, alpha = self.model.backbone.get_alpha_values(
                    seasonal.permute(0, 2, 1),
                    trend.permute(0, 2, 1),
                )                                           # alpha: [B, C, 1]
                all_alphas.append(alpha.squeeze(-1).mean(dim=0).cpu())
        return torch.stack(all_alphas).mean(dim=0)         # [C]

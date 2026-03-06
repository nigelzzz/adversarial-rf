"""
Adversarial Training: train model on a mix of clean and adversarial examples.

Each batch: generate PGD adversarial examples, then:
    loss = alpha * CE(model(x_adv), y) + (1-alpha) * CE(model(x_clean), y)
"""

import os
import time

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from util.adv_attack import Model01Wrapper
from util.sigguard_eval import create_attack, generate_adversarial
from util.early_stop import EarlyStopping
from util.logger import AverageMeter


class AdversarialTrainer:
    """
    Adversarial training with mixed clean/adversarial loss.

    Args:
        model: Classification model
        train_loader: Training data loader
        val_loader: Validation data loader
        cfg: Configuration object
        logger: Logger instance
        adv_alpha: Weight for adversarial loss (0..1), clean weight = 1-alpha
        adv_attack: Attack name for generating adversarial examples
        adv_eps: Epsilon for the attack
        model_name: Model architecture name
    """

    def __init__(self, model, train_loader, val_loader, cfg, logger,
                 adv_alpha=0.5, adv_attack='pgd', adv_eps=0.03,
                 model_name='awn'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.logger = logger
        self.adv_alpha = adv_alpha
        self.adv_attack_name = adv_attack
        self.adv_eps = adv_eps
        self.model_name = model_name.upper()
        self.device = cfg.device

        self.ta_box = str(getattr(cfg, 'ta_box', 'unit')).lower()

        # Will be initialized in loop()
        self.optimizer = None
        self.criterion = None
        self.early_stopping = None

    def _create_attack(self):
        """Create attack object for adversarial example generation."""
        wrapped = Model01Wrapper(self.model)
        wrapped.to(self.device)
        # Temporarily store original eps, override for adv training
        orig_eps = getattr(self.cfg, 'attack_eps', 0.03)
        self.cfg.attack_eps = self.adv_eps
        atk = create_attack(self.adv_attack_name, wrapped, self.cfg)
        self.cfg.attack_eps = orig_eps
        return atk, wrapped

    def loop(self):
        """Main training loop."""
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.early_stopping = EarlyStopping(self.logger, patience=self.cfg.patience)

        best_val_acc = 0.0
        lr_list = []
        train_loss_list, train_acc_list = [], []
        val_loss_list, val_acc_list = [], []

        for epoch in range(self.cfg.epochs):
            # --- Train ---
            self.model.train()
            train_loss_meter = AverageMeter()
            train_acc_meter = AverageMeter()
            t_start = time.time()

            # Create fresh attack each epoch (model weights change)
            atk, wrapped = self._create_attack()

            with tqdm(total=len(self.train_loader),
                      desc=f'AdvTrain {epoch}/{self.cfg.epochs}',
                      mininterval=0.3) as pbar:
                for sig_batch, lab_batch in self.train_loader:
                    sig_batch = sig_batch.to(self.device)
                    lab_batch = lab_batch.to(self.device)

                    # Generate adversarial examples
                    self.model.eval()
                    with torch.no_grad():
                        pass  # ensure eval mode for attack
                    try:
                        x_adv = generate_adversarial(
                            atk, sig_batch, lab_batch,
                            wrapped_model=wrapped,
                            ta_box=self.ta_box,
                            fallback_to_single=False,
                        )
                    except Exception:
                        x_adv = sig_batch  # fallback to clean on failure

                    # Mixed forward pass
                    self.model.train()
                    logit_clean, regu_clean = self.model(sig_batch)
                    logit_adv, regu_adv = self.model(x_adv)

                    loss_clean = self.criterion(logit_clean, lab_batch)
                    loss_adv = self.criterion(logit_adv, lab_batch)

                    # Add regularization from both
                    regu = sum(regu_clean) + sum(regu_adv)

                    loss = ((1 - self.adv_alpha) * loss_clean
                            + self.adv_alpha * loss_adv
                            + regu)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Track clean accuracy
                    preds = logit_clean.argmax(dim=1)
                    acc = (preds == lab_batch).float().mean().item()
                    train_loss_meter.update(loss.item())
                    train_acc_meter.update(acc)

                    pbar.set_postfix(loss=train_loss_meter.avg,
                                     acc=train_acc_meter.avg)
                    pbar.update(1)

            lr_list.append(self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_loss_meter.avg)
            train_acc_list.append(train_acc_meter.avg)
            self.logger.info(
                f'Epoch {epoch}: Train Loss={train_loss_meter.avg:.4f}, '
                f'Train Acc={train_acc_meter.avg:.4f}, '
                f'Time={time.time()-t_start:.1f}s')

            # --- Validate ---
            self.model.eval()
            val_loss_meter = AverageMeter()
            val_acc_meter = AverageMeter()

            with torch.no_grad():
                for sig_batch, lab_batch in self.val_loader:
                    sig_batch = sig_batch.to(self.device)
                    lab_batch = lab_batch.to(self.device)
                    logit, regu_sum = self.model(sig_batch)
                    loss = self.criterion(logit, lab_batch) + sum(regu_sum)
                    preds = logit.argmax(dim=1)
                    acc = (preds == lab_batch).float().mean().item()
                    val_loss_meter.update(loss.item())
                    val_acc_meter.update(acc)

            val_loss_list.append(val_loss_meter.avg)
            val_acc_list.append(val_acc_meter.avg)
            self.logger.info(
                f'Epoch {epoch}: Val Loss={val_loss_meter.avg:.4f}, '
                f'Val Acc={val_acc_meter.avg:.4f}')

            # Save best model
            if val_acc_meter.avg >= best_val_acc:
                best_val_acc = val_acc_meter.avg
                save_name = f"{self.cfg.dataset}_{self.model_name}_ADVTRAIN.pkl"
                torch.save(self.model.state_dict(),
                           os.path.join(self.cfg.model_dir, save_name))
                self.logger.info(f"Saved best model: {save_name} "
                                 f"(val_acc={best_val_acc:.4f})")

            # Early stopping
            self.early_stopping(val_loss_meter.avg, self.model)
            if self.early_stopping.early_stop:
                self.logger.info('Early stopping')
                break

            if (self.early_stopping.counter != 0
                    and self.early_stopping.counter % self.cfg.milestone_step == 0):
                lr = self.optimizer.param_groups[0]['lr'] * self.cfg.gamma
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr

        self.logger.info(f"Adversarial training complete. Best val acc: "
                         f"{best_val_acc:.4f}")
        return best_val_acc

import torch
from torch import nn
from  torch import optim
from torch.nn import utils as torch_utils

from tqdm import tqdm
from src import utils
from math import exp

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

class Trainer(object):

    def __init__(self, model, crit, config, **kwargs):
        self.model = model
        self.crit = crit
        self.config = config

        super(Trainer, self).__init__()

        self.n_epochs = config.n_epochs
        self.lower_is_better = True
        self.best = {
            'epoch': 0,
            'current_lr': config.lr,
            'config': config,
            **kwargs
        }

    def train(self,
              train,
              valid,
              verbose=VERBOSE_EPOCH_WISE):

        current_lr = self.best['current_lr']
        best_loss = float('Inf') * (1 if self.lower_is_better else -1)
        lowest_after = 0

        progress_bar = tqdm(range(self.best['epoch'], self.n_epochs),
                            desc='Training: ',
                            unit='epoch') if verbose is VERBOSE_EPOCH_WISE else range(self.best['epoch'], self.n_epochs)

        if self.best['epoch'] > 0:
            avg_valid_loss = self.validate(valid, verbose=verbose)

        for idx in progress_bar:

            if self.config.adam:
                optimizer = optim.Adam(self.model.parameters(), lr=current_lr)
            else:
                optimizer = optim.SGD(self.model.parameters(), lr=current_lr)
            print('current learning rate: ', current_lr)

            if verbose > VERBOSE_EPOCH_WISE:
                print('epoch: %d/%d\tmin_valid_loss=%.4e' % (idx + 1,
                                                             self.n_epochs,
                                                             best_loss
                                                             ))
            avg_train_loss, avg_param_norm, avg_grad_norm = self.train_epoch(train, optimizer, verbose=verbose)
            avg_valid_loss = self.validate(valid, verbose=verbose)

            if verbose is VERBOSE_EPOCH_WISE:
                progress_bar.set_postfix_str('|param|=%.2f |g_param|=%.2f train_loss=%.4e valid_loss=%.4e min_valid_loss=%.4e' % (avg_param_norm,
                                                                                                                                  avg_grad_norm,
                                                                                                                                  avg_train_loss,
                                                                                                                                  avg_valid_loss,
                                                                                                                                  best_loss
                                                                                                                                  ))
            if (self.lower_is_better and avg_valid_loss < best_loss) or (not self.lower_is_better and avg_valid_loss > best_loss):
                best_loss = avg_valid_loss
                lowest_after = 0

                self.best['model'] = self.model.state_dict()
                self.best['optim'] = optimizer
                self.best['epoch'] = idx + 1
                self.best['current_lr'] = current_lr

                model_fn = self.config.model.split('.')
                if self.config.n_epochs >= idx + 1:
                    model_fn = model_fn[:-1] + ['%02d' % (idx + 1),
                                                '%.2f-%.2f' % (avg_train_loss,
                                                               exp(avg_train_loss)
                                                               ),
                                                '%.2f-%.2f' % (avg_valid_loss,
                                                               exp(avg_valid_loss)
                                                               )
                                                ] + [model_fn[-1]]
                else:
                    model_fn = model_fn[:-1] + ['%02d' % (idx + 1),
                                                '%.2f' % (avg_train_loss),
                                                '%.2f' % (avg_valid_loss)
                                                ] + [model_fn[-1]]
                self.save_training('.'.join(model_fn))
                else:
                lowest_after += 1

                if lowest_after >= self.config.early_stop and  self.config.early_stop > 0:
                    break

            # Altough there is an improvement in last epoch, we need to decay the learning-rate if it meets the requirements.
            if ((lowest_after > 0) or (idx + 1 >= self.config.lr_decay_start_at)) and (idx + 1 <= self.config.n_epochs):
                current_lr = max(self.config.min_lr, current_lr * self.config.lr_decay_rate)
        if verbose is VERBOSE_EPOCH_WISE:
            progress_bar.close()


    def validate(self,
                 valid,
                 crit=None,
                 verbose=VERBOSE_BATCH_WISE):

        with torch.no_grad():
            sample_cnt = 0
            total_loss, total_word_count = 0, 0

            progress_bar = tqdm(valid,
                                desc='Validation: ',
                                unit='batch') if verbose is VERBOSE_BATCH_WISE else valid
            self.model.eval()
            for idx, mini_batch in enumerate(progress_bar):
                x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
                # |x| = (batch_size, length)
                # |y| = (batch_size, length)
                y_hat = self.model(x, mini_batch.tgt[0][:, :-1])
                # |y_hat| = (batch_size, n_classes)
                loss = self._get_loss(y_hat, y, crit)

                total_loss += float(loss)
                total_word_count += int(mini_batch.tgt[1].sum())
                avg_loss = total_loss / total_word_count

                sample_cnt += mini_batch.tgt[0].size(0)

                if verbose is VERBOSE_BATCH_WISE:
                    progress_bar.set_postfix_str('valid loss=%.4e PPL=%.2f' % (avg_loss, exp(avg_loss)))

                if idx >= len(progress_bar):
                    break

            self.model.train()

            if verbose is VERBOSE_BATCH_WISE:
                progress_bar.close()

            return avg_loss

    def train_epoch(self,
                    train,
                    optimizer,
                    verbose=VERBOSE_BATCH_WISE):
        total_loss, total_word_count = 0, 0
        total_grad_norm = 0
        avg_loss, avg_grad_norm = 0, 0
        sample_cnt = 0

        if verbose == VERBOSE_BATCH_WISE:
            print(optimizer)

        progress_bar = tqdm(train,
                            desc='Training: ',
                            unit='batch') if verbose is VERBOSE_BATCH_WISE else train

        for idx, mini_batch in enumerate(progress_bar):
            x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
            # |x| = (batch_size, length)
            # |y| = (batch_size, length)
            optimizer.zero_grad()
            y_hat = self.model(x, mini_batch.tgt[0][:, :-1])
            # |y_hat| = (batch_size, length, output_size)
            loss = self._get_loss(y_hat, y)
            loss.div(y.size(0)).backward()

            total_loss += float(loss)
            total_word_count += int(mini_batch.tgt[1].sum())
            param_norm = float(utils.get_parameter_norm(self.model.parameters()))
            total_grad_norm += float(utils.get_grad_norm(self.model.parameters()))

            avg_loss = total_loss / total_word_count
            avg_grad_norm = total_grad_norm / (idx + 1)

            if verbose is VERBOSE_BATCH_WISE:
                progress_bar.set_postfix_str('|param|=%.2f |g_param|=%.2f loss=%.4e PPL=%.2f' % (param_norm,
                                                                                                 avg_grad_norm,
                                                                                                 avg_loss,
                                                                                                 exp(avg_loss)
                                                                                                 ))

            # In orther to avoid gradient exploding, we apply gradient clipping.
            torch_utils.clip_grad_norm_(self.model.parameters(),
                                        self.config.max_grad_norm
                                        )
            # Take a step of gradient descent.
            optimizer.step()

            sample_cnt += mini_batch.tgt[0].size(0)

            if idx >= len(progress_bar) * self.config.train_ratio_per_epoch:
                break

        if verbose is VERBOSE_BATCH_WISE:
            progress_bar.close()

        return avg_loss, param_norm, avg_grad_norm

    def get_best_model(self):
        self.model.load_state_dict(self.best['model'])
        return self.model

    def save_training(self, fn):
        torch.save(self.best, fn)

    def _get_loss(self, y_hat, y, crit=None):
        # |y_hat| = (batch_size, length, output_size)
        # |y| = (batch_size, length)
        crit = self.crit if crit is None else crit
        loss = crit(y_hat.contiguous().view(-1, y_hat.size(-1)),
                    y.contiguous().view(-1))
        return loss


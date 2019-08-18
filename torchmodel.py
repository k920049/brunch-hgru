import torch

from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor

from fire import Fire
from tqdm import tqdm
import numpy as np

from model.torch.Wrapper import HierarchicalRNN
from model.torch.Loss import SampledSoftmax
from pipeline.torch.Dataset import BrunchDataset, generate_batches


class HierarchicalRecommender(object):

    def __init__(self,
                 epoch=10,
                 batch_size=128,
                 history_length=128,
                 hidden_size=256,
                 nsampled=4,
                 bptt_step=128,
                 optimizer="adam",
                 pretrained_weight="./data/brunch/embedding.npy",
                 model_path="./data/checkpoints/torch/best_model.ckpt"):

        self.epoch = epoch
        self.batch_size = batch_size
        self.history_length = history_length
        self.bptt_step = bptt_step
        self.optimizer = optimizer
        self.model_path = model_path

        self.pretrained_weight = np.load(pretrained_weight)
        self.ntokens = self.pretrained_weight.shape[0]
        self.embedding_dim = self.pretrained_weight.shape[1]
        self.hidden_size = hidden_size

        self.train_dataset = BrunchDataset()

        self.embedding = torch.nn.Embedding.from_pretrained(torch.FloatTensor(self.pretrained_weight))

        self.rnn = HierarchicalRNN(word_vec_dim=self.embedding_dim,
                                   hidden_size=hidden_size,
                                   session_num_layers=2,
                                   user_num_layers=2,
                                   dropout_p=0.1,
                                   bptt_step=bptt_step)

        self.sampled_softmax = SampledSoftmax(ntokens=self.ntokens,
                                              nsampled=nsampled,
                                              nhid=hidden_size,
                                              tied_weight=None)

        parameters = [elem for elem in self.rnn.parameters()] + [elem for elem in self.sampled_softmax.parameters()]
        self.optimizer = optim.Adam(params=parameters,
                                    lr=0.001)

    def train(self):
        try:
            best_loss = float("Inf")

            for idx in range(1, self.epoch + 1):
                self.train_epoch(idx)
                loss = self.eval_epoch(idx)

                if loss < best_loss:
                    with open(self.model_path, 'wb') as f:
                        model_state = {
                            'cell': self.cell,
                            'sampled_softmax': self.sampled_softmax,
                        }
                        torch.save(model_state, f)
                    best_loss = loss

        except KeyboardInterrupt:
            print('Exiting from training early')

    def train_epoch(self, epoch):
        """
         Train one more epoch
        :return: Nothing
        """
        self.train_dataset.set_split("train")
        self.embedding.train()
        self.rnn.train()
        self.sampled_softmax.train()
        progress_bar = tqdm(generate_batches(self.train_dataset,
                                             batch_size=self.batch_size))

        for batch_idx, batch_dict in enumerate(progress_bar):

            self.optimizer.zero_grad()
            total_loss = 0

            session_input = batch_dict["session_input"].long()
            session_output = batch_dict["session_output"].long()
            session_mask = batch_dict["session_mask"].float()
            user_mask = batch_dict["user_mask"].float()
            mask = batch_dict["mask"].float()

            session_state = None
            user_state = None

            for step in range(0, session_input.size(1), self.bptt_step):
                session_state = self.repackage_hidden(session_state)
                user_state = self.repackage_hidden(user_state)

                input_embedding = self.embedding(session_input[:, step: step + self.bptt_step])
                output, session_state, user_state = self.rnn(input_embedding,
                                                             session_state=session_state,
                                                             user_state=user_state,
                                                             session_mask=session_mask[:, step: step + self.bptt_step],
                                                             user_mask=user_mask[:, step: step + self.bptt_step])
                # compute the top-1 ranking loss
                current_session_output = session_output[:, step: step + self.bptt_step].contiguous()
                curent_mask = mask[:, step: step + self.bptt_step].contiguous()
                logits, new_targets = self.sampled_softmax(output.view(-1, output.size(2)), current_session_output.view(-1))

                true_logit = logits[:, 0].unsqueeze(1)
                sampled_logit = logits[:, 1:]
                loss = torch.sigmoid(sampled_logit - true_logit) + torch.sigmoid(torch.mul(sampled_logit, sampled_logit))
                loss = torch.mean(loss, dim=1)
                loss = torch.mul(loss, curent_mask.view(-1))
                loss = torch.sum(loss)

                loss.backward()

                torch.nn.utils.clip_grad_norm(self.rnn.parameters(), 0.25)
                torch.nn.utils.clip_grad_norm(self.sampled_softmax.parameters(), 0.25)
                self.optimizer.step()

                total_loss = total_loss + loss.data

            message = '| Train | epoch {:3d} | {:5d}/{:5d} batches | loss {:5.2f} |'.format(
                epoch,
                batch_idx * self.batch_size,
                len(self.train_dataset),
                total_loss)
            progress_bar.set_description(message)

    def eval_epoch(self, epoch):
        """
         Evaluate what we have learned
        :return: loss
        """

        self.train_dataset.set_split("valid")
        self.embedding.eval()
        self.rnn.eval()
        self.sampled_softmax.eval()

        total_loss = 0
        idx = 0

        for batch_idx, batch_dict in enumerate(generate_batches(self.train_dataset, batch_size=self.batch_size)):

            idx = batch_idx

            session_input = pad_sequence(batch_dict["session_input"], batch_first=True)
            session_output = pad_sequence(batch_dict["session_output"], batch_first=True)
            session_mask = pad_sequence(batch_dict["session_mask"], batch_first=True)
            user_mask = pad_sequence(batch_dict["user_mask"], batch_first=True)
            mask = pad_sequence(batch_dict["mask"], batch_first=True)

            session_state = None
            user_state = None

            for step in range(0, session_input.size(1), self.bptt_step):
                input_embedding = self.embedding(session_input[:, step + self.bptt_step])
                output, session_state, user_state = self.rnn(input_embedding,
                                                             session_state=session_state,
                                                             user_state=user_state,
                                                             session_mask=session_mask[:, step + self.bptt_step],
                                                             user_mask=user_mask[:, step + self.bptt_step])
                # compute the top-1 ranking loss
                session_output = session_output[:, step: step + self.bptt_step].contiguous()
                mask = mask[:, step: step + self.bptt_step].contiguous()
                logits, new_targets = self.sampled_softmax(output.view(-1, output.size(2)), session_output.view(-1))
                true_logit = logits[:, 0]
                sampled_logit = logits[:, 1:]
                loss = torch.sigmoid(sampled_logit - true_logit) + torch.sigmoid(torch.mul(sampled_logit, sampled_logit))
                loss = torch.mean(loss, dim=1)
                loss = torch.mul(loss, mask.view(-1))
                loss = torch.sum(loss)

                total_loss = total_loss + loss

        message = '| Evaluation | epoch {:3d} | loss {:5.2f} |'.format(epoch, total_loss / idx)
        print(message)
        return total_loss

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if h is None:
            return h
        if type(h) == Tensor:
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)


if __name__ == "__main__":
    # Fire(HierarchicalRecommender)
    HierarchicalRecommender().train()

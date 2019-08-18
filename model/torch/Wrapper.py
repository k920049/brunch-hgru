import torch

from torch import nn
from model.torch.Cell import HierarchicalRNNCell


class HierarchicalRNN(nn.Module):

    def __init__(self,
                 word_vec_dim,
                 hidden_size,
                 session_num_layers=2,
                 user_num_layers=2,
                 dropout_p=0.1,
                 bptt_step=32):

        super(HierarchicalRNN, self).__init__()

        self.word_vec_dim = word_vec_dim
        self.hidden_size = hidden_size
        self.session_num_layers = session_num_layers
        self.user_num_layers = user_num_layers
        self.dropout_p = dropout_p
        self.bptt_step = bptt_step

        self.cell = HierarchicalRNNCell(word_vec_dim=word_vec_dim,
                                        hidden_size=hidden_size,
                                        sesssion_num_layers=session_num_layers,
                                        user_num_layers=user_num_layers,
                                        dropout_p=dropout_p)

    def forward(self, input, session_state, user_state, session_mask, user_mask):
        """
         Step forward self.bptt_step amount of time
        :param input: (batch size, bptt step, embedding dim)
        :param session_state: [(batch size, bptt step, hidden size)] * session num layers
        :param user_state: [(batch size, bptt step, hidden size)] * user num layers
        :param session_mask: (batch size, bptt step, 1)
        :param user_mask: (batch size, bptt step, 1)
        :return:
            output: (batch size, bptt step, hidden size)
            new session state: [(batch size, bptt step, hidden size)] * session num layers
            new user state: user_state: [(batch size, bptt step, hidden size)] * user num layers
        """

        if session_state is None:
            session_state, _ = self._initial_hidden(input.size(0))

        if user_state is None:
            _, user_state = self._initial_hidden(input.size(0))

        assert len(
            session_state) == self.session_num_layers, "The number of session layers and the number of session states don't match."
        assert len(
            user_state) == self.user_num_layers, "The number of user layers and the number of user states don't match."
        assert input.size(
            1) == self.bptt_step, "Input time step doesn't match with the amount of step you set in bptt_step"

        output_list = []

        for step in range(input.size(1)):
            current_input = input[:, step, :]
            current_session_mask = session_mask[:, step].unsqueeze(1)
            current_user_mask = user_mask[:, step].unsqueeze(1)

            output, session_state, user_state = self.cell(current_input,
                                                          session_state=session_state,
                                                          user_state=user_state,
                                                          session_mask=current_session_mask,
                                                          user_mask=current_user_mask)

            output_list.append(output.unsqueeze(1))

        output = torch.cat(output_list, dim=1)
        return output, session_state, user_state

    def _initial_hidden(self, batch_size):
        session_state = [torch.zeros((batch_size, self.hidden_size)) for _ in range(self.session_num_layers)]
        user_states = [torch.zeros((batch_size, self.hidden_size)) for _ in range(self.user_num_layers)]

        return session_state, user_states

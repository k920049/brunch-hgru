import torch

from torch import nn
from torch.nn import GRUCell
from model.torch.ListModule import ListModule


class HierarchicalRNNCell(nn.Module):

    def __init__(self,
                 word_vec_dim,
                 hidden_size,
                 sesssion_num_layers=2,
                 user_num_layers=2,
                 dropout_p=0.1):
        super(HierarchicalRNNCell, self).__init__()

        self.word_vec_dim = word_vec_dim
        self.hidden_size = hidden_size
        self.session_num_layers = sesssion_num_layers
        self.user_num_layers = user_num_layers
        self.dropout_p = dropout_p


        self.session_rnn_cells = [GRUCell(input_size=word_vec_dim, hidden_size=hidden_size)]
        self.session_rnn_cells = self.session_rnn_cells + [
            GRUCell(input_size=hidden_size, hidden_size=hidden_size) for _ in range(sesssion_num_layers - 1)
        ]
        self.session_rnn_cells = ListModule(*self.session_rnn_cells)

        self.user_rnn_cells = [
            GRUCell(input_size=hidden_size, hidden_size=hidden_size) for _ in range(user_num_layers)
        ]
        self.user_rnn_cells = ListModule(*self.user_rnn_cells)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, input, session_state, user_state, session_mask, user_mask):
        """
         One step forward
        :param input: (batch_size, word_vec_dim)
        :param session_state: [(batch_size, hidden_size)] * session_num_layers
        :param user_state: [(batch_size, hidden_size)] * user_num_layers
        :param session_mask: (batch_size, 1)
        :param user_mask: (batch_size, 1)
        :return: output, new_session_state, new_user_state
        """

        assert len(self.session_rnn_cells) == len(self.user_rnn_cells),\
            "The number of user cells and session cells doesn't match"

        new_session_state = []
        new_user_state = []
        session_mask = session_mask
        user_mask = user_mask

        for idx in range(self.session_num_layers):

            prev_session_state  = session_state[idx]
            prev_user_state     = user_state[idx]

            session_state = torch.mul(session_mask, prev_session_state) + torch.mul(1.0 - session_mask, prev_user_state)
            input = self.session_rnn_cells[idx](input, session_state)
            new_session_state.append(input)

            next_session_state = self.user_rnn_cells[idx](input, prev_user_state)
            next_session_state = torch.mul(user_mask, next_session_state) + torch.mul(1.0 - user_mask, prev_session_state)
            new_user_state.append(next_session_state)

        output = self.dropout(input)

        return output, new_session_state, new_user_state

    def _initial_hidden(self, batch_size):
        session_state   = [torch.zeros((batch_size, self.hidden_size)) for _ in range(self.session_num_layers)]
        user_states     = [torch.zeros((batch_size, self.hidden_size)) for _ in range(self.user_num_layers)]

        return session_state, user_states

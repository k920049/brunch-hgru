import tensorflow as tf

from model.tensorflow.Helper import _generate_zero_filled_state_for_cell, _is_multiple_state
from tensorflow.python.keras.layers import deserialize as deserialize_layer
from tensorflow.python.keras.layers.recurrent import *


class HierarchicalRNNCell(Layer):

    def __init__(self,
                 user_cells,
                 session_cells,
                 embedding_layer,
                 **kwargs):

        self.user_cells         = self.verify_cells(user_cells)
        self.session_cells      = self.verify_cells(session_cells)
        self.user_cell_size     = self.user_cells[0].state_size
        self.session_cell_size  = self.session_cells[0].state_size
        self.cells = self.session_cells + self.user_cells
        self.embedding = embedding_layer

        super(HierarchicalRNNCell, self).__init__(**kwargs)

    def call(self, inputs, states, constants=None, **kwargs):
        # Recover per-cell states.
        state_size = (self.state_size)
        nested_states = nest.pack_sequence_as(state_size, nest.flatten(states))

        sequence        = tf.gather(inputs, indices=[0], axis=1)            # [batch_size, 1]
        session_mask    = tf.gather(inputs, indices=[1], axis=1)            # [batch_size, 1]
        user_mask       = tf.gather(inputs, indices=[2], axis=1)            # [batch_size, 1]
        session_inputs  = tf.squeeze(self.embedding(sequence), axis=1)      # [batch_size, embedding_dim]

        new_nested_states = []
        for idx, (cell_s, cell_u) in enumerate(zip(self.session_cells, self.user_cells)):
            # state of the previous session cell
            state_s = nested_states[2 * idx]
            # state of the previous user cell
            state_u = nested_states[2 * idx + 1]
            # mask the previous session cell
            state_s = tf.multiply(session_mask, state_s) + tf.multiply(1.0 - session_mask, state_u)
            # run the session cell
            if generic_utils.has_arg(cell_s.call, "constants"):
                session_inputs, new_state_s = cell_s.call(session_inputs, [state_s], constants=constants, **kwargs)
            else:
                session_inputs, new_state_s = cell_s.call(session_inputs, [state_s], **kwargs)
            new_nested_states.append(new_state_s[0])

            user_inputs = new_state_s[0]
            if generic_utils.has_arg(cell_u.call, "constants"):
                user_inputs, new_state_u = cell_u.call(user_inputs, [state_u], constants=constants, **kwargs)
            else:
                user_inputs, new_state_u = cell_u.call(user_inputs, [state_u], **kwargs)
            new_state_u = tf.multiply(user_mask, new_state_u[0]) + tf.multiply(1.0 - user_mask, state_u)
            new_nested_states.append(new_state_u)

        return session_inputs, nest.pack_sequence_as(state_size, nest.flatten(new_nested_states))


    @tf_utils.shape_type_conversion
    def build(self, input_shape):

        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        input_shape = (input_shape[0], self.embedding.output_dim)
        for idx, (cell_s, cell_u) in enumerate(zip(self.session_cells, self.user_cells)):
            with tf.name_scope("session-{}".format(idx)):
                # Setting a session cell first
                if isinstance(cell_s, Layer):
                    if not cell_s.built:
                        cell_s.build(input_shape)

                if getattr(cell_s, 'output_size', None) is not None:
                    output_dim = cell_s.output_size
                elif _is_multiple_state(cell_s.state_size):
                    output_dim = cell_s.state_size[0]
                else:
                    output_dim = cell_s.state_size
            with tf.name_scope("user-{}".format(idx)):
                input_shape = tuple([input_shape[0]] + tensor_shape.as_shape(output_dim).as_list())
                # Then set a user cell later
                if isinstance(cell_u, Layer):
                    if not cell_u.built:
                        cell_u.build(input_shape)
        self.built = True

    @staticmethod
    def verify_cells(cells):
        for cell in cells:
            if not hasattr(cell, 'call'):
                raise ValueError('All cells must have a `call` method. received cells:', cells)
            if not hasattr(cell, 'state_size'):
                raise ValueError('All cells must have a `state_size` attribute. received cells:', cells)
        return cells

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self.cells)

    @property
    def output_size(self):
        if getattr(self.session_cells[-1], 'output_size', None) is not None:
            return self.session_cells[-1].output_size
        elif _is_multiple_state(self.session_cells[-1].state_size):
            return self.session_cells[-1].state_size[0]
        else:
            return self.session_cells[-1].state_size

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        initial_states = []
        for cell in self.cells:
            initial_states.append(_generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype))
        return tuple(initial_states)


    def get_config(self):
        user_cells = []
        for cell in self.user_cells:
            user_cells.append({
                'class_name': cell.__class__.__name__,
                'config': cell.get_config()
            })
        session_cells = []
        for cell in self.session_cells:
            session_cells.append({
                'class_name': cell.__class__.__name__,
                'config': cell.get_config()
            })
        embedding_config = {
            'class_name': self.embedding.__class__.__name__,
            'config': self.embedding.get_config()
        }
        config = {
            'user_cells': user_cells,
            'session_cells': session_cells,
            'embedding_layer': embedding_config
        }
        base_config = super(HierarchicalRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):

        user_cells = []
        session_cells = []

        for cell_config in config.pop('user_cells'):
            user_cells.append(deserialize_layer(cell_config, custom_objects=custom_objects))
        for cell_config in config.pop('session_cells'):
            session_cells.append(deserialize_layer(cell_config, custom_objects=custom_objects))
        embedding = deserialize_layer(config.pop('embedding_layer'), custom_objects=custom_objects)

        return cls(user_cells, session_cells, embedding, **config)


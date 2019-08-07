from tensorflow.python.keras.layers.recurrent import *

def _is_multiple_state(state_size):
    """Check whether the state_size contains multiple states."""
    return (hasattr(state_size, '__len__') and not isinstance(state_size, tensor_shape.TensorShape))

def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
    if inputs is not None:
        batch_size = array_ops.shape(inputs)[0]
        dtype = inputs.dtype
    return _generate_zero_filled_state(batch_size, cell.state_size, dtype)

def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
    """Generate a zero filled tensor with shape [batch_size, state_size]."""
    if batch_size_tensor is None or dtype is None:
        raise ValueError('batch_size and dtype cannot be None while constructing initial state: batch_size={}, dtype={}'.format(batch_size_tensor, dtype))

    def create_zeros(unnested_state_size):
        flat_dims = tensor_shape.as_shape(unnested_state_size).as_list()
        init_state_size = [batch_size_tensor] + flat_dims
        return array_ops.zeros(init_state_size, dtype=dtype)

    if nest.is_sequence(state_size):
        return nest.map_structure(create_zeros, state_size)
    else:
        return create_zeros(state_size)
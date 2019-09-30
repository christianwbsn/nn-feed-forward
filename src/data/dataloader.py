"""
Module implementing data loading and splitting
"""

import numpy as np

def mini_batch(inputs, targets, batch_size, shuffle=False):
    """
    Return the mini batch for each defined inputs and targets

    Takes:
        inputs, a `List` of input/features to learned.
        targets, a `List` of gold/target
        batch_size, `int` which determines batch size
        shuffle, `bool` default=False
    Returns a `tuple` indicates list of inputs and targets
    """
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
        end_idx = min(start_idx + batch_size, inputs.shape[0])
        if shuffle:
            excerpt = indices[start_idx : end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield inputs[excerpt], targets[excerpt]
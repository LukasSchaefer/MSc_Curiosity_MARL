import numpy as np

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def masked_max(actions, action_mask):
    """
    Apply masked softmax function
    :param actions: actor network output
    :param action_mask: mask of binary values in action output size
    :return: max_masked_actions, max_index_masked_actions
    """
    if action_mask is not None:
        mask = torch.from_numpy(np.array(action_mask)).float().to(device)
        while mask.dim() < actions.dim():
            mask = mask.unsqueeze(1)
        actions_masked = actions * action_mask
        return actions_masked.max(1)
    else:
        return actions.max(1)


from collections import OrderedDict


def load_checkpoints(model, weight_state_dict, logger):
    if 'state_dict' in weight_state_dict.keys():
        model.load_state_dict(weight_state_dict["state_dict"], strict=True)
        logger.info(">>> Loaded weights from pretrained checkpoint")
    else:
        # deepspeed
        new_weight_state_dict = OrderedDict()
        for key in weight_state_dict['module'].keys():
            new_weight_state_dict[key] = weight_state_dict['module'][key]
        model.load_state_dict(new_weight_state_dict, strict=True)
        logger.info(">>> Loaded weights from pretrained checkpoint (Deepspeed)")

    return model
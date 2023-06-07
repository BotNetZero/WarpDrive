import torch

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def create_optimizer(model, optimizer_type, weight_decay=0.01, learning_rate=2e-5,
                     adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-6):

    if optimizer_type == 'adamw' or optimizer_type == 'adam':
        from torch.optim import AdamW
        print('>>>>> using Adam')
    elif optimizer_type == '8bit-adam':
        from bitsandbytes.optim import Adam8bit as AdamW
        print('>>>>> using 8bit-Adam')
    else:
        assert False

    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [
        name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters and p.requires_grad],
            "weight_decay": 0.0,
        }
    ]
    optimizer_cls = AdamW
    optimizer_kwargs = {
        "betas": (adam_beta1, adam_beta2),
        "eps": adam_epsilon,
    }
    optimizer_kwargs["lr"] = learning_rate
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer

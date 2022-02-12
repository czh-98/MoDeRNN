from .MoDeRNN import *


def get_convrnn_model(name, **kwargs):
    models = {
        'MoDeRNN': get_MoDeRNN,
    }
    return models[name](**kwargs)

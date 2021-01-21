from .deeplabv3_plus import *
from .danet import *

__all__ = ['get_model', 'get_model_list', 'get_segmentation_model']

# Code used for prediction
_models = {
    'deeplabv3_plus_xception_voc': get_deeplabv3_plus_xception_voc,
    'danet_resnet50_ciyts': get_danet_resnet50_citys,
    'danet_resnet101_citys': get_danet_resnet101_citys,
    'danet_resnet152_citys': get_danet_resnet152_citys,
}


def get_model(name, **kwargs):
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net


def get_model_list():
    return _models.keys()


def get_segmentation_model(model, **kwargs):
    models = {
        # 'fcn32s': get_fcn32s,
        # 'fcn16s': get_fcn16s,
        # 'fcn8s': get_fcn8s,
        # 'fcn': get_fcn,
        # 'psp': get_psp,
        # 'deeplabv3': get_deeplabv3,
        'deeplabv3_plus': get_deeplabv3_plus,
        'danet': get_danet,
        # 'denseaspp': get_denseaspp,
        # 'bisenet': get_bisenet,
        # 'encnet': get_encnet,
        # 'dunet': get_dunet,
        # 'icnet': get_icnet,
        # 'enet': get_enet,
        # 'ocnet': get_ocnet,
        # 'ccnet': get_ccnet,
        # 'psanet': get_psanet,
        # 'cgnet': get_cgnet,
        # 'espnet': get_espnet,
        # 'lednet': get_lednet,
        # 'dfanet': get_dfanet,
        # 'unet' : get_unet,
        # 'yanannet' : get_mynet,
        # 'rcaanet' : get_mynet,
    }
    return models[model](**kwargs)

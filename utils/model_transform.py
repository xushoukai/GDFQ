import torch.nn as nn
import torch
import numpy as np

__all__ = ["data_parallel", "model2list",
           "list2sequential", "model2state_dict"]


def data_parallel(model, ngpus, gpu0=0):
    """
    assign model to multi-gpu mode
    :params model: target model
    :params ngpus: number of gpus to use
    :params gpu0: id of the master gpu
    :return: model, type is Module or Sequantial or DataParallel
    """
    if ngpus == 0:
        assert False, "only support gpu mode"
    gpu_list = list(range(gpu0, gpu0 + ngpus))
    assert torch.cuda.device_count() >= gpu0 + ngpus, "Invalid Number of GPUs"
    if isinstance(model, list):
        for i in range(len(model)):
            if ngpus >= 2:
                if not isinstance(model[i], nn.DataParallel):
                    model[i] = torch.nn.DataParallel(model[i], gpu_list).cuda()
            else:
                model[i] = model[i].cuda()
    else:
        if ngpus >= 2:
            if not isinstance(model, nn.DataParallel):
                model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()
    return model


def model2list(model):
    """
    convert model to list type
    :param model: should be type of list or nn.DataParallel or nn.Sequential
    :return: no return params
    """
    if isinstance(model, nn.DataParallel):
        model = list(model.module)
    elif isinstance(model, nn.Sequential):
        model = list(model)
    return model


def list2sequential(model):
    if isinstance(model, list):
        model = nn.Sequential(*model)
    return model


def model2state_dict(file_path):
    model = torch.load(file_path)
    if model['model'] is not None:
        model_state_dict = model['model'].state_dict()
        torch.save(model_state_dict, file_path.replace(
            '.pth', 'state_dict.pth'))

    else:
        print((type(model)))
        print(model)
        print("skip")

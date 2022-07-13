import os
import yaml
import datetime
from typing import Union, Any, Dict
from importlib import import_module


class Solver(object):
    def __init__(self, opt, dev):
        self.opt = opt
        self.dev = dev

        # make main folder
        if not os.path.exists(opt.ExperimentName):
            os.makedirs(opt.ExperimentName)

        with open(opt.ExperimentName + '/config.txt', 'a') as f:
            f.write(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '\n\n')
            write_option(f, opt)
            f.write('\n')

    def execute(self):
        raise NotImplementedError()


class Options(object):
    def __init__(self, data: dict):
        self.dir_list = []
        for key, value in data.items():
            setattr(self, key, self._warp(value))
            self.dir_list.append(key)

    def __dir__(self):
        return self.dir_list

    def _warp(self, data: Union[Dict, Any]):
        if isinstance(data, dict):
            return Options(data)
        else:
            return data


def write_option(f, opt: Options, pretext=''):
    for key in dir(opt):
        value = getattr(opt, key)
        if isinstance(value, Options):
            f.write(f'{pretext}{key}:\n')
            write_option(f, value, pretext=pretext+'    ')
        else:
            f.write(f'{pretext}{key}: {value}\n')


def get_options():
    with open('options.yml', 'r', encoding='utf8') as f:
        opt = yaml.load(f, Loader=yaml.Loader)

        model_name = opt['Model']
        if os.path.exists('./Model/{}/config.yml'.format(model_name)):
            with open('./Model/{}/config.yml'.format(model_name), 'r', encoding='utf8') as f:
                model_opt = yaml.load(f, Loader=yaml.Loader)
            opt[model_name] = model_opt
        else:
            raise FileNotFoundError("There is no config file in './Model/{}' directory".format(model_name))
    opt = Options(opt)
    return opt


def get_policy(opt, dev):
    model_name = opt.Model
    module = import_module('Model.{}.policy'.format(model_name))
    policy = module.Policy(opt, dev)
    return policy

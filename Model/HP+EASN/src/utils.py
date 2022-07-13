import sys
import time
import logging
import numpy as np
from collections import OrderedDict


def write_bitstream(f, bitstream_dict):
    strings = bitstream_dict['strings']
    shape = bitstream_dict['shape']
    
    # save strings
    for idx in range(2):
        f.write(np.array(len(strings[idx][0]), dtype=np.int32).tobytes())
        f.write(strings[idx][0])

    # save shape
    f.write(np.array(shape[0], dtype=np.int32).tobytes())
    f.write(np.array(shape[1], dtype=np.int32).tobytes())


def read_bitstream(f):
    strings = list()
    shape = list()

    # read strings
    for _ in range(len(strings)):
        length_ = np.frombuffer(f.read(4), dtype=np.int32)[0]   # read 4 bytes
        strings.append([f.read(length_)])
    for _ in range(len(shape)):
        shape_0 = np.frombuffer(f.read(4), dtype=np.int32)[0]
        shape_1 = np.frombuffer(f.read(4), dtype=np.int32)[0]
        shape.append([shape_0, shape_1])
    return {'strings': strings, 'shape': shape}


class ElapsedTimeProcess(object):
    def __init__(self, max_epoch: int, max_step_per_epoch: int, gamma: float=0.99):
        self.max_epoch = max_epoch
        self.max_step_per_epoch = max_step_per_epoch

        self.gamma = gamma
        self.time_step = 0.0

        self.t1 = 0
        self.t2 = 0

    def start(self):
        self.t1 = time.time()

    def end(self, current_epoch, current_step):
        self.t2 = time.time()

        time_step = self.t2 - self.t1
        if self.time_step == 0.0:
            self.time_step = time_step
        else:
            self.time_step = self.time_step * self.gamma + time_step * (1 - self.gamma)

        total_elapsed_step = (self.max_step_per_epoch - current_step - 1) + self.max_step_per_epoch * (self.max_epoch - current_epoch)
        eta = self.time_step * total_elapsed_step

        elapsed_time_dict = self._calculate_summary(eta)
        return self._to_string(elapsed_time_dict)

    @staticmethod
    def _calculate_summary(eta):
        elapsed_time_dict = OrderedDict()

        # days
        eta_days = int(eta // (24 * 3600))
        if eta_days != 0:
            elapsed_time_dict['eta_days'] = eta_days

        # hours
        eta_hours = int((eta // 3600) % 24)
        if eta_hours != 0:
            elapsed_time_dict['eta_hours'] = eta_hours

        # minutes
        eta_minutes = int((eta // 60) % 60)
        if eta_minutes != 0:
            elapsed_time_dict['eta_minutes'] = eta_minutes

        # seconds
        elapsed_time_dict['eta_seconds'] = int(eta % 60)

        return elapsed_time_dict

    @staticmethod
    def _to_string(elapsed_time_dict):
        output = ''
        for key, value in elapsed_time_dict.items():
            if key == 'eta_days':
                output += '{} days '.format(value)
            elif key == 'eta_hours':
                output += '{} h '.format(value)
            elif key == 'eta_minutes':
                output += '{} m '.format(value)
            elif key == 'eta_seconds':
                output += '{} s'.format(value)
            else:
                raise KeyError('Some key has mismatched name')
        return output


class ProcessBar(object):
    def __init__(self, max_iter, prefix='', suffix='', bar_length=50):
        self.max_iter = max_iter
        self.prefix = prefix
        self.suffix = suffix
        self.bar_length = bar_length
        self.iteration = 0

    def step(self, iteration=None, other_info: str=None):
        if iteration is None:
            self.iteration += 1

        percent = 100 * self.iteration / self.max_iter
        filled_length = int(round(self.bar_length * self.iteration) / self.max_iter)
        bar = '#' * filled_length + '-' * (self.bar_length - filled_length)
        msg = '\r{} [{}] {:.1f}% {}'.format(self.prefix, bar, percent, self.suffix)
        if other_info is not None:
            msg = msg + "  |   " + other_info
        sys.stdout.write(msg)
        if self.iteration == self.max_iter:
            sys.stdout.write('\n')
        sys.stdout.flush()


class Logger(object):
    def __init__(self, logging_file_dir):
        self.logger = logging.getLogger('log')
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler(logging_file_dir)
        self.logger.addHandler(handler)

    def __call__(self, msg):
        self.logger.info(msg)
        print(msg)


class MovingAverageMeter(object):
    def __init__(self, gamma=0.95):
        self.gamma = gamma
        self.buckets = None

    def update(self, log: dict):
        if self.buckets is None:
            self.buckets = log
        else:
            for key, value in log.items():
                self.buckets[key] = (1 - self.gamma) * value + self.gamma * self.buckets[key]

    def get_value(self):
        return self.buckets



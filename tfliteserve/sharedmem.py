"""
'Host' a model.TFLiteModel using a shared memory buffer

the shared buffer needs to be sized to handle:
    input tensor
    output tensor (after quantization removal)
    status [1 byte?]
        0 idle [set by client, after output is read]
        1 input ready [set by client]
        2 processing [set by server]
        3 output ready [set by server]
"""

import base64
import logging
import os
import pickle
import time

import numpy

from . import model


# names of mmap files (/dev/shm is shared memory)
default_folder = '/dev/shm/tfliteserver'
in_array_fn = 'nn_input'
out_array_fn = 'nn_output'
status_fn = 'nn_status'
meta_fn = 'meta'

STATUS_IDLE = 0
STATUS_INPUT_READY = 1
STATUS_PROCESSING = 2
STATUS_OUTPUT_READY = 3

status_value_to_name = {
    STATUS_IDLE: 'idle',
    STATUS_INPUT_READY: 'input ready',
    STATUS_PROCESSING: 'processing',
    STATUS_OUTPUT_READY: 'output ready',
}
status_name_to_value = {v: k for k, v in status_value_to_name.items()}


class SharedMem:
    def __init__(self, meta=None, folder=None):
        if folder is None:
            folder = default_folder
        self.folder = folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        if meta is None:
            # load meta from directory
            with open(os.path.join(folder, meta_fn), 'rb') as f:
                meta = pickle.load(f)
        else:
            # write meta to directory
            with open(os.path.join(folder, meta_fn), 'wb') as f:
                pickle.dump(meta, f)

        if 'input' not in meta:
            raise ValueError("Invalid metadata missing input")

        if 'output' not in meta:
            raise ValueError("Invalid metadata missing output")
        
        if 'labels' not in meta:
            meta['labels'] = {}

        input_shape = tuple(meta['input']['shape'])
        input_dtype = numpy.dtype(meta['input']['dtype'])

        self.input_array = numpy.memmap(
            os.path.join(folder, in_array_fn),
            dtype=input_dtype, mode='w+', shape=input_shape)

        # if output is quantized, it's dtype won't match output_details
        output_shape = tuple(meta['output']['shape'])
        if 'quantization' in meta['output']:
            output_dtype = numpy.dtype('f8')
        else:
            output_dtype = numpy.dtype(meta['output']['dtype'])

        self.output_array = numpy.memmap(
            os.path.join(folder, out_array_fn),
            dtype=output_dtype, mode='w+', shape=output_shape)
        
        # status file/pipe?
        self.status_array = numpy.memmap(
            os.path.join(folder, status_fn),
            dtype=numpy.uint8, mode='w+', shape=(1, ))

    def set_input(self, input_array):
        self.input_array[:] = input_array
        self.input_array.flush()

    def get_input(self, copy=False):
        if copy:
            return numpy.copy(self.input_array)
        return self.input_array

    def set_output(self, output_array):
        self.output_array[:] = output_array
        self.output_array.flush()

    def get_output(self, copy=False):
        if copy:
            return numpy.copy(self.output_array)
        return self.output_array

    def get_status(self, parse=False):
        if not parse:
            return int(self.status_array[0])
        return self.parse_status()
    
    def set_status(self, code):
        if isinstance(code, str):
            code = status_name_to_value[code]
        self.status_array[0] = code
        self.status_array.flush()
    
    def parse_status(self):
        return status_value_to_name[self.status_array[0]]

    def wait_for_status(self, code, timeout=None, delay=None):
        if isinstance(code, str):
            code = status_name_to_value[code]
        if delay is None:
            wait = lambda: None
        else:
            wait = lambda: time.sleep(delay)
        if timeout is None:
            should_exit = lambda: False
        else:
            t0 = time.monotonic()
            should_exit = lambda: ((time.monotonic() - t0) > timeout)
        while self.status_array[0] != code:
            wait()
            if should_exit():
                return False
        return True


class SharedMemoryServer:
    def __init__(self, model, poll_delay=0.001, folder=None):
        self.model = model
        meta = {
            'input': model.input_details,
            'output': model.output_details,
            'labels': model.labels,
        }
        self.buffers = SharedMem(meta, folder=None)
        self.buffers.set_status(STATUS_IDLE)
        self.poll_delay = poll_delay

    def run(self):
        self.buffers.set_status(STATUS_PROCESSING)
        self.buffers.set_output(self.model.run(self.buffers.input_array))
        self.buffers.set_status(STATUS_OUTPUT_READY)

    def run_forever(self):
        while True:
            if self.buffers.get_status() == STATUS_INPUT_READY:
                self.run()
            else:
                time.sleep(self.poll_delay)


class SharedMemoryClient:
    def __init__(self, folder=None):
        self.buffers = SharedMem(folder=folder)

    def run(self, input_array):
        # TODO add timeouts, delays
        self.buffers.wait_for_status(STATUS_IDLE)
        self.buffers.set_input(input_array)
        self.buffers.set_status(STATUS_INPUT_READY)
        self.buffers.wait_for_status(STATUS_OUTPUT_READY)
        o = self.buffers.get_output(copy=True)
        self.buffers.set_status(STATUS_IDLE)
        return o

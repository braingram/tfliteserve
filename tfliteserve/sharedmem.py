"""
'Host' a model.TFLiteModel using shared memory buffers

the shared buffers include: 
    input array
    output array (after quantization removal)
    status [1 byte]
        0 idle [set by client, after output is read]
        1 input ready [set by client]
        2 processing [set by server]
        3 output ready [set by server]

Need separate codes per client, clients will need to have a unique channel.
Channel negotiation is... complicated, skip it, just pre-assign.

Each client should make it's own folder

main_dir: /dev/shm/tfliteserver

each client directory has:
    - status
    - input
    - output
    - meta: input/output details

client connects:
    - makes new folder in directory (if directory exists, delete it)
    - server sees new folder, fills with meta data & buffers
    - client connects to buffers
"""

import base64
import logging
import os
import pickle
import shutil
import time

import numpy

from . import model


default_server_folder = '/dev/shm/tfliteserver'

STATUS_IDLE = 0
STATUS_INPUT_READY = 1
STATUS_PROCESSING = 2
STATUS_OUTPUT_READY = 3
STATUS_CLIENT_PING = 10

status_code_to_name = {
    STATUS_IDLE: 'idle',
    STATUS_INPUT_READY: 'input ready',
    STATUS_PROCESSING: 'processing',
    STATUS_OUTPUT_READY: 'output ready',
}
status_name_to_code = {v: k for k, v in status_code_to_name.items()}


class SharedMemoryBuffers:
    """
    Construct shared buffers (or access existing ones) in a folder on disk.
    These buffers are designed to be used by multiple processes where
    one process (Server) handles 'processing' input provided from other
    processes (Clients) via the input_array shared buffer and results
    are returned from the Server to the Client via the output_array.
    
    Coordination of the shared buffers occures via transmission of codes
    via the status_array. See get_status for a description of codes.

    Processing proceeds as follows:
        TODO

    Parameters
    ---------
    name: string
        Buffer collection name (will be subdir in shared buffer server_folder).
    meta: dict or None
        Dictionary containing metadata about the shared buffers including:
            input: dict, see tensorflow flite interpreter input_details
            output: dict, see tensorflow flite interpreter output_details
            labels: dict, see model.load_labels
        If None, meta will be read from the meta file in the folder
    server_folder: string or None
        Folder path in which to store shared memory mapped files. If None
        default_server_folder will be used.
    create: bool, default=True
        Create shared buffers otherwise remove folder and wait for another
        process to create the buffers.

    Attributes
    ----------
    folder: string
        Folder containing shared memory mapped files.
    input_array: numpy.memmap
        Shape = meta['input']['shape'], dtype = meta['input]['dtype']
    output_array: numpy.memmap
        Shape = meta['input']['shape'], dtype = meta['input]['dtype']
        if model has quantization, dtype will be overridden as float64
    status_array: numpy.memmap
        single element, uint8 array containing a status byte that
        defines when the input or output is safe to read or write
        see get_status

    Methods
    -------
    set_input: Copy values into input array shared buffer.
    get_input: Get contents or reference of input array.
    set_output: Copy values into output array shared buffer. 
    get_output: Get contents or reference of output array.
    get_status: Get current status code or string.
    set_status: Set current status code or string.
    wait_for_status: Wait until status changes to a provided code.

    Notes
    -----
    Direct use of arrays (without copying values) is very discouraged as
    these memmapped values can be changed by other processes. Instead,
    use wait_for_status to know when it is safe to read or write an array
    using the get/set_input/output methods.

    Examples
    --------
    """
    def __init__(self, name, meta=None, server_folder=None, create=True):
        if server_folder is None:
            server_folder = default_server_folder
        folder = os.path.join(server_folder, name)
        self.folder = folder
        if create:
            if meta is None:
                raise ValueError("meta is required if creating buffers")
            self._create_buffers(meta)
        else:
            self._wait_for_buffers()

    def _create_buffers(self, meta=None):
        mfn = os.path.join(self.folder, 'meta')
        if meta is None:
            # load meta from directory
            with open(mfn, 'rb') as f:
                meta = pickle.load(f)
        else:
            # write meta to directory
            with open(mfn, 'wb') as f:
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
            os.path.join(self.folder, 'input'),
            dtype=input_dtype, mode='w+', shape=input_shape)

        # if output is quantized, it's dtype won't match output_details
        output_shape = tuple(meta['output']['shape'])
        if 'quantization' in meta['output']:
            output_dtype = numpy.dtype('f8')
        else:
            output_dtype = numpy.dtype(meta['output']['dtype'])

        self.output_array = numpy.memmap(
            os.path.join(self.folder, 'output'),
            dtype=output_dtype, mode='w+', shape=output_shape)
        
        # status file/pipe?
        self.status_array = numpy.memmap(
            os.path.join(self.folder, 'status'),
            dtype=numpy.uint8, mode='w+', shape=(1, ))

    def _wait_for_buffers(self):
        # if folder exists, remove it
        if os.path.exists(self.folder):
            shutil.rmtree(self.folder)
            while os.path.exists(self.folder):
                time.sleep(0.001)
        os.makedirs(self.folder)
        sfn = os.path.join(self.folder, 'status')
        while not os.path.exists(sfn):
            time.sleep(0.001)
        self._create_buffers()

    def set_input(self, values):
        """Copy values to input_array buffer and flush to disk

        Parameters
        ----------
        values: ndarray
            Must be the same size and dtype of input_array
        """
        self.input_array[:] = values
        self.input_array.flush()

    def get_input(self, copy=True):
        """Get a copy of or reference to input_array

        Parameters
        ----------
        copy: bool, default=True
            if True, copy contents of input_array, if False a reference
            to input_array will be returned. Use of the reference should
            be restricted to times when get_status returns a safe code.
            see get_status for more information.

        Returns
        -------
        values: ndarray
            Either a copy or reference of output_array.
        """
        if copy:
            return numpy.copy(self.input_array)
        return self.input_array

    def set_output(self, values):
        """Copy values to output_array buffer and flush to disk

        Parameters
        ----------
        values: ndarray
            Must be the same size and dtype of output_array
        """
        self.output_array[:] = values
        self.output_array.flush()

    def get_output(self, copy=True):
        """Get a copy of or reference to output_array

        Parameters
        ----------
        copy: bool, default=True
            if True, copy contents of output_array, if False a reference
            to output_array will be returned. Use of the reference should
            be restricted to times when get_status returns a safe code.
            see get_status for more information.

        Returns
        -------
        values: ndarray
            Either a copy or reference of input_array.
        """
        if copy:
            return numpy.copy(self.output_array)
        return self.output_array

    def get_status(self, parse=False):
        """Get the current status code or string

        Status codes and allowable acces to input and output arrays are:
            0: idle
                No buffers are being processed, it is safe to write to the
                input_array and then set the status to 'input ready'
                    input_array: rw
                    output_array: rw
            1: input ready
                New values were written to input array and are awaiting
                processing. No access is permitted.
                    input_array: -
                    output_array: -
            2: processing
                Current buffers are being processed. No access is permitted.
                    input_array: -
                    output_array: -
            3: output ready
                New output was generated it should be read from output_array
                and the status set to idle
                    input_array: rw
                    output_array: rw

        Parameters
        ----------
        parse: bool, default=False
            If True, lookup and return the string description of the status

        Returns
        -------
        status: int or string
            Current status code or description (see parse above)
        """
        if parse:
            return status_value_to_name[self.status_array[0]]
        return int(self.status_array[0])
    
    def set_status(self, code):
        """Set the current status code

        Parameters
        ----------
        code: int or string
            Status code or description (see get_status)
        """
        if isinstance(code, str):
            code = status_name_to_code[code]
        self.status_array[0] = code
        self.status_array.flush()
    
    def wait_for_status(self, code, timeout=None, delay=None):
        """Wait until status changes to the provided code

        Parameters
        ----------
        code: int or string
            Status code or description (see get_status) to wait for.
        timeout: float or None
            If None, function blocks until status matches code. If float,
            wait for a maximum of timeout seconds.
        delay: float or None
            If None, blocking polls status with no delay. If float, wait
            for delay seconds between checks of status code.

        Returns
        -------
        status_reached: bool
            True if status code matched, False if timed out before code
            matched.
        """
        if isinstance(code, str):
            code = status_name_to_code[code]
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
    def __init__(self, model, poll_delay=0.0001, server_folder=None):
        self.model = model
        if server_folder is None:
            server_folder = default_server_folder
        self.folder = server_folder
        self.poll_delay = poll_delay
        self.clients = {}
        self.meta = {
            'input': model.input_details,
            'output': model.output_details,
            'labels': model.labels,
        }

    def check_for_new_clients(self):
        for cn in os.listdir(self.folder):
            if cn not in self.clients:
                self.add_client(cn)
            else:
                # check for status file
                sfn = os.path.join(self.folder, cn, 'status')
                if not os.path.exists(sfn):
                    self.add_client(cn)

    def add_client(self, name):
        self.clients[name] = SharedMemoryBuffers(
            name, self.meta, server_folder=self.folder, create=True)
        self.clients[name].set_status(STATUS_IDLE)

    def check_clients(self):
        r = False
        for cn in self.clients:
            b = self.clients[cn]
            if b.get_status() == STATUS_INPUT_READY:
                b.set_status(STATUS_PROCESSING)
                b.set_output(self.model.run(b.input_array))
                b.set_status(STATUS_OUTPUT_READY)
                r = True

    def run_forever(self):
        while True:
            if not self.check_clients():
                self.check_for_new_clients()  # TODO only do this periodically
                time.sleep(self.poll_delay)


class SharedMemoryClient:
    def __init__(self, name, server_folder=None):
        self.name = name
        self.buffers = SharedMemoryBuffers(
            name, server_folder=server_folder, create=False)

    def run(self, input_array, timeout=None, delay=None):
        if not self.buffers.wait_for_status(
                STATUS_IDLE, timeout=timeout, delay=delay):
            return None
        self.buffers.set_input(input_array)
        self.buffers.set_status(STATUS_INPUT_READY)
        if not self.buffers.wait_for_status(
                STATUS_OUTPUT_READY, timeout=timeout, delay=delay):
            return None
        o = self.buffers.get_output(copy=True)
        self.buffers.set_status(STATUS_IDLE)
        return o

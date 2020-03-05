"""
'Host' a model.TFLiteModel using shared memory buffers

the shared buffers include: 
    input array
    output array (after quantization removal)

Need separate codes per client, clients will need to have a unique channel.
Channel negotiation is... complicated, skip it, just pre-assign.

Each client should make it's own folder

main_dir: /dev/shm/tfliteserver

each client directory has:
    - input
    - output
    - meta: input/output details

client connects:
    - makes new folder in directory (if directory exists, delete it)
    - server sees new folder, fills with meta data & buffers
    - client connects to buffers
"""

import asyncio
import os
import pickle
import select
import shutil
import time

import numpy


default_server_folder = '/dev/shm/tfliteserver'


def validate_meta(meta):
    if 'input' not in meta:
        raise ValueError("Invalid metadata missing input")

    if 'output' not in meta:
        raise ValueError("Invalid metadata missing output")

    for k in ('input', 'output'):
        if k not in meta:
            raise ValueError("Invalid metadata missing %s" % k)
        for sk in ('shape', 'dtype'):
            if sk not in meta[k]:
                raise ValueError("Invalid metadata missing %s %s" % (k, sk))


class SharedMemoryBuffers:
    """
    Construct shared buffers (or access existing ones) in a folder on disk.
    These buffers are designed to be used by multiple processes where
    one process (Server) handles 'processing' input provided from other
    processes (Clients) via the input_array shared buffer and results
    are returned from the Server to the Client via the output_array.
    
    Coordination of the shared buffers occures via two fifos.


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

    Methods
    -------
    set_input: Copy values into input array shared buffer.
    get_input: Get contents or reference of input array.
    set_output: Copy values into output array shared buffer. 
    get_output: Get contents or reference of output array.

    Notes
    -----
    Direct use of arrays (without copying values) is very discouraged as
    these memmapped values can be changed by other processes.

    Examples
    --------
    """
    def __init__(self, name, meta=None, server_folder=None, create=True):
        if server_folder is None:
            server_folder = default_server_folder
        folder = os.path.join(server_folder, name)
        self.folder = folder
        self.is_client = None
        if create:
            if meta is None:
                raise ValueError("meta is required if creating buffers")
            self._create_buffers(meta)
        else:
            self._wait_for_buffers()

    def _create_buffers(self, meta=None):
        mfn = os.path.join(self.folder, 'meta')

        ir_fn = os.path.join(self.folder, 'input_ready')  # to server
        or_fn = os.path.join(self.folder, 'output_ready')  # to client

        if meta is None:
            self.is_client = True
            # load meta from directory
            with open(mfn, 'rb') as f:
                meta = pickle.load(f)
        else:
            self.is_client = False
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)
            for fn in (ir_fn, or_fn):
                if not os.path.exists(fn):
                    os.mkfifo(fn)
            # write meta to directory
            print("Saving meta to %s" % mfn)
            with open(mfn, 'wb') as f:
                pickle.dump(meta, f)

        validate_meta(meta)
        self.meta = meta

        input_shape = tuple(meta['input']['shape'])
        input_dtype = numpy.dtype(meta['input']['dtype'])

        self.input_array = numpy.memmap(
            os.path.join(self.folder, 'input'),
            dtype=input_dtype, mode='w+', shape=input_shape)

        output_shape = tuple(meta['output']['shape'])
        output_dtype = numpy.dtype(meta['output']['dtype'])

        self.output_array = numpy.memmap(
            os.path.join(self.folder, 'output'),
            dtype=output_dtype, mode='w+', shape=output_shape)
        
        if self.is_client:
            self.output_ready_fp = open(or_fn, 'r')
            self.input_ready_fp = open(ir_fn, 'w')
            ffp = self.output_ready_fp
        else:
            self.output_ready_fp = open(or_fn, 'w')
            self.input_ready_fp = open(ir_fn, 'r')
            ffp = self.input_ready_fp

        # flush input
        while ffp in select.select([ffp,], [], [], 0.001)[0]:
            ffp.read(1)

    def __del__(self):
        self.input_ready_fp.close()
        self.output_ready_fp.close()

    def _wait_for_buffers(self):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        fns = [
            os.path.join(self.folder, fn) for fn in [
                'meta',
                'input',
                'output',
                'input_ready',
                'output_ready']
        ]
        for fn in fns:
            print("Waiting for %s" % fn)
            while not os.path.exists(fn):
                time.sleep(0.001)
        self._create_buffers()

    def set_input(self, values):
        """Copy values to input_array buffer and flush to disk

        Parameters
        ----------
        values: ndarray
            Must be the same size and dtype of input_array
        """
        assert self.is_client
        self.input_array[:] = values
        self.input_array.flush()

        self.input_ready_fp.write("1")
        self.input_ready_fp.flush()

    def get_input(self, copy=True):
        """Get a copy of or reference to input_array

        Parameters
        ----------
        copy: bool, default=True
            if True, copy contents of input_array, if False a reference
            to input_array will be returned.

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
        assert not self.is_client
        self.output_array[:] = values
        self.output_array.flush()

        self.output_ready_fp.write("1")
        self.output_ready_fp.flush()

    def get_output(self, copy=True):
        """Get a copy of or reference to output_array

        Parameters
        ----------
        copy: bool, default=True
            if True, copy contents of output_array, if False a reference
            to output_array will be returned.

        Returns
        -------
        values: ndarray
            Either a copy or reference of input_array.
        """
        if copy:
            return numpy.copy(self.output_array)
        return self.output_array


class SharedMemoryServer:
    def __init__(self, function, meta, server_folder=None):
        self.function = function
        if server_folder is None:
            server_folder = default_server_folder
        self.folder = server_folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.clients = {}
        validate_meta(meta)
        self.meta = meta

        # make junk data
        #self.junk_input = numpy.random.randint(
        #    0, 255, size=self.meta['input']['shape'],
        #    dtype=self.meta['input']['dtype'])

    def check_for_new_clients(self):
        for cn in os.listdir(self.folder):
            if cn not in self.clients:
                self.add_client(cn)
            # else:  # TODO check if client is still alive, read if not
        self.loop.call_later(
            1.0, self.check_for_new_clients)

    def add_client(self, name):
        if name in self.clients:
            del self.clients[name]
        self.clients[name] = SharedMemoryBuffers(
            name, self.meta, server_folder=self.folder, create=True)
        self.loop.add_reader(
            self.clients[name].input_ready_fp, self.run_client, name)

    def run_client(self, client_name):
        print("Running client: %s" % client_name)
        b = self.clients[client_name]
        b.input_ready_fp.read(1)
        b.set_output(self.function(b.input_array))

    #def run_junk(self):
    #    self.function(self.junk_input)
    #    self.loop.call_later(
    #        0.010, self.run_junk)

    def run_forever(self, loop=None):
        if loop is None:
            loop = asyncio.get_event_loop()
        
        self.loop = loop
        self.loop.call_soon(self.check_for_new_clients)
        
        #self.loop.call_soon(self.run_junk)

        self.loop.run_forever()
        self.loop.close()


class SharedMemoryClient:
    def __init__(self, name, server_folder=None):
        self.name = name
        self.buffers = SharedMemoryBuffers(
            name, server_folder=server_folder, create=False)

    def run(self, input_array, timeout=None, delay=None):
        self.buffers.set_input(input_array)

        # TODO better integrate this with the buffers
        self.buffers.output_ready_fp.read(1)
        return self.buffers.get_output(copy=True)

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

import argparse

from . import model
from . import sharedmem


def run_server():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--edge', default=False, action='store_true',
        help='run model on the edge')
    parser.add_argument(
        '-f', '--fake', default=False, action='store_true',
        help='run fake model')
    parser.add_argument(
        '-j', '--junk', default=0.01, type=float,
        help='period to inject junk data to keep edge tpu happy')
    parser.add_argument(
        '-l', '--labels', default='',
        help='labels filename [if blank model must be directory]')
    parser.add_argument(
        '-m', '--model', default='',
        help='Model directory or file')
    args = parser.parse_args()
    if args.fake:
        meta = {
            'input': {'shape': (1, 224, 224, 3), 'dtype': 'uint8'},
            'output': {'shape': (1, 1024), 'dtype': 'f8'},
        }

        def f(in_array):
            out_array = numpy.zeros(
                meta['output']['shape'], dtype=meta['output']['dtype'])
            out_array[:] = in_array.mean()
            return out_array

        s = sharedmem.SharedMemoryServer(f, meta)
    else:
        m = model.TFLiteModel(args.model, args.labels, edge=args.edge)
        s = model.TFLiteServer(m, junk_period=args.junk)
    s.run_forever()

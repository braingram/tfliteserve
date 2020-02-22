import argparse

from . import model
from . import sharedmem


def run_server():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--edge', default=False, action='store_true',
        help='run model on the edge')
    parser.add_argument(
        '-l', '--labels', default='', required=True,
        help='labels filename [if blank model must be directory]')
    parser.add_argument(
        '-m', '--model', default='', required=True,
        help='Model directory or file')
    parser.add_argument(
        '-p', '--poll_delay', default=0.001, type=float,
        help='polling delay [seconds] for server checking status')
    args = parser.parse_args()

    m = model.TFLiteModel(args.model, args.labels, edge=args.edge)
    s = sharedmem.SharedMemoryServer(m, m.meta, poll_delay=args.poll_delay)
    s.run_forever()

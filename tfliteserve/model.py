import time

import numpy

import tflite_runtime.interpreter as tflite
load_delegate = tflite.load_delegate


EDGETPU_SHARED_LIB = 'libedgetpu.so.1'


# load labels
def load_labels(path, encoding='utf-8'):
    """Loads labels from file (with or without index numbers).

    Args:
        path: path to label file.
        encoding: label file encoding.
    Returns:
        Dictionary mapping indices to labels.
    """
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}

        if lines[0].split(' ', maxsplit=1)[0].isdigit():
            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}


class TFLiteModel:
    def __init__(self, model_fn, labels_fn=None, edge=False):
        # load model
        if edge:
            self.model = tflite.Interpreter(
                  model_path=model_fn,
                  experimental_delegates=[load_delegate(EDGETPU_SHARED_LIB, {})])
        else:
            self.model = tflite.Interpreter(model_path=model_fn)
        self.model.allocate_tensors()

        # load labels (if provided)
        if labels_fn is not None:
            self.labels = load_labels(labels_fn)
        else:
            self.labels = {}

        # prep input details
        self.input_details = self.model.get_input_details()[0]
        self.output_details = self.model.get_output_details()[0]
        self.input_tensor_index = self.input_details['index']
        self.output_tensor_index = self.output_details['index']
        self.q_out_scale, self.q_out_zero = self.output_details['quantization']

        self.meta = {
            'input': self.input_details,
            'output': self.output_details,
            'labels': self.labels,
        }
        if 'quantization' in self.meta['output']:
            self.meta['output']['dtype'] = 'f8'

    def set_input(self, input_tensor):
        if input_tensor.dtype != self.input_details['dtype']:
            print("Converting input dtype:", input_tensor.dtype, self.input_details['dtype'])
            input_tensor = input_tensor.astype(self.input_details['dtype'])
        if (
                input_tensor.ndim != len(self.input_details['shape']) or
                numpy.any(input_tensor.shape != self.input_details['shape'])):
            print("Reshaping input:", input_tensor.shape, self.input_details['shape'])
            input_tensor = input_tensor.reshape(self.input_details['shape'])
        self.model.set_tensor(self.input_tensor_index, input_tensor)

    def get_output(self):
        return (
            self.q_out_scale *
            (
                numpy.squeeze(self.model.get_tensor(self.output_tensor_index)) -
                self.q_out_zero))

    def run(self, input_tensor):
        self.set_input(input_tensor)
        self.model.invoke()
        return self.get_output()

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

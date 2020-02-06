You might want to just use one of the official Tensorflow model servers

This tool serves a tflite model (optionally running on an accelerator) over
a shared memory interface (memmap in /dev/shm).

To run server
```bash

python3 -m tfliteserve -m tflite_model_filename -l labels_filename

# optionally: -e runs the model with an edge_tpu accelerator
```

To use the server
```python

import tfliteserve.sharedmem

c = tfliteserve.sharedmem.SharedMemoryClient()

results = c.run(image)
# where image is a correctly sized and typed numpy array
```

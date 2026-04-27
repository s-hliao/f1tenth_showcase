"""
trt_infer.py

TensorRT inference wrapper for DAv2 ViT-S.

Drop-in replacement for the PyTorch model in depth_fusion_node.py:
    inferencer = TRTInferencer("checkpoints/depth_anything_v2_vits.engine")
    depth_np = inferencer.infer_image(bgr_image)   # same as model.infer_image()

The engine must have been built with export_trt.py (fixed 518×518 input, fp16).
Preprocessing (BGR→RGB, normalize, resize to 518×518) is done here in numpy/cv2
to match the original DAv2 image2tensor pipeline.

CUDA is initialised lazily on the first infer_image() call so that the context
lives on whichever thread actually runs inference — avoids cross-thread issues.
"""

import numpy as np
import cv2

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 518


class TRTInferencer:
    def __init__(self, engine_path: str):
        self._engine_path = engine_path
        self._ready = False   # True after _init_cuda() runs on the inference thread

    # ------------------------------------------------------------------
    def _init_cuda(self):
        """Initialise CUDA + TRT on the calling thread (done once, lazily)."""
        import pycuda.driver as cuda
        import tensorrt as trt

        cuda.init()
        self._cuda_ctx = cuda.Device(0).make_context()

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self._engine_path, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

        ctx = engine.create_execution_context()

        n = engine.num_io_tensors
        names = [engine.get_tensor_name(i) for i in range(n)]
        input_name  = next(nm for nm in names
                           if engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT)
        output_name = next(nm for nm in names
                           if engine.get_tensor_mode(nm) == trt.TensorIOMode.OUTPUT)

        in_shape  = (1, 3, INPUT_SIZE, INPUT_SIZE)
        out_shape = (1, INPUT_SIZE, INPUT_SIZE)

        h_in  = cuda.pagelocked_empty(int(np.prod(in_shape)),  dtype=np.float32)
        h_out = cuda.pagelocked_empty(int(np.prod(out_shape)), dtype=np.float32)
        d_in  = cuda.mem_alloc(h_in.nbytes)
        d_out = cuda.mem_alloc(h_out.nbytes)
        stream = cuda.Stream()

        ctx.set_tensor_address(input_name,  int(d_in))
        ctx.set_tensor_address(output_name, int(d_out))

        self._cuda   = cuda
        self._ctx    = ctx
        self._h_in   = h_in
        self._h_out  = h_out
        self._d_in   = d_in
        self._d_out  = d_out
        self._stream = stream
        self._ready  = True

    # ------------------------------------------------------------------
    def _preprocess(self, raw_image: np.ndarray) -> np.ndarray:
        """BGR HxWx3 uint8 → float32 CHW normalised, resized to INPUT_SIZE×INPUT_SIZE."""
        img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE),
                         interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD          # HxWx3
        return img.transpose(2, 0, 1)       # CHW

    # ------------------------------------------------------------------
    def infer_image(self, raw_image: np.ndarray, input_size: int = INPUT_SIZE) -> np.ndarray:
        """
        Same interface as DepthAnythingV2.infer_image().
        Returns a float32 disparity map at the original image resolution.
        """
        if not self._ready:
            self._init_cuda()

        h, w = raw_image.shape[:2]

        chw = self._preprocess(raw_image)
        np.copyto(self._h_in, chw.ravel())

        cuda = self._cuda
        cuda.memcpy_htod_async(self._d_in, self._h_in, self._stream)
        self._ctx.execute_async_v3(self._stream.handle)
        cuda.memcpy_dtoh_async(self._h_out, self._d_out, self._stream)
        self._stream.synchronize()

        depth = self._h_out.reshape(INPUT_SIZE, INPUT_SIZE)

        if (h, w) != (INPUT_SIZE, INPUT_SIZE):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        return depth

    # ------------------------------------------------------------------
    def close(self):
        if self._ready:
            self._cuda_ctx.pop()
            self._cuda_ctx.detach()
            self._ready = False

import onnxruntime
import numpy as np

device_name = 'cpu'
print(onnxruntime.get_available_providers())

if device_name == 'cpu':
    providers = ['CPUExecutionProvider']
elif device_name =='cuda:0':
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# 创建session
# onnx_model = onnxruntime.InferenceSession('../runs/train/yolov7-tiny/weights/best.ONNX', providers=providers)
onnx_model = onnxruntime.InferenceSession('../weights/yolov7-tiny.onnx', providers=providers)
# create the input
data = np.random.rand(1, 3, 640, 640).astype(np.float32)
# inference
onnx_input = {onnx_model.get_inputs()[0].name: data}
outputs = onnx_model.run(None, onnx_input)
print(len(outputs))

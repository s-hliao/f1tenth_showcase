import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import glob
import time

CLASS_NAMES = ['car'] 
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

class F110Detection:
    def __init__(self, engine_path, conf_thres=0.4, iou_thres=0.5):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.allocations = []
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            
            nptype = np.dtype(trt.nptype(dtype))
            host_mem = np.zeros(shape, dtype=nptype)
            allocation = cuda.mem_alloc(int(trt.volume(shape) * nptype.itemsize))
            self.allocations.append(int(allocation))
            
            binding = {
                'name': name,
                'dtype': nptype,
                'shape': list(shape),
                'allocation': allocation,
                'host_mem': host_mem
            }
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

    def preprocess(self, img):
        input_h, input_w = self.inputs[0]['shape'][2], self.inputs[0]['shape'][3]
        img_resized = cv2.resize(img, (input_w, input_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        img_chw = img_float.transpose(2, 0, 1)
        return np.ascontiguousarray(img_chw[None, ...])

    def detect(self, frame):
        input_data = self.preprocess(frame)
        cuda.memcpy_htod(self.inputs[0]['allocation'], input_data)
        
        self.context.execute_v2(self.allocations)
        
        output_binding = self.outputs[0]
        cuda.memcpy_dtoh(output_binding['host_mem'], output_binding['allocation'])
        
        return self.post_process(frame, output_binding['host_mem'])

    def post_process(self, origin_img, trt_output):
        predictions = np.squeeze(trt_output)
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(1, -1)

        input_h, input_w = self.inputs[0]['shape'][2], self.inputs[0]['shape'][3]
        img_h, img_w = origin_img.shape[:2]
        
        boxes = predictions[:, :4] * input_w
        box_scores = predictions[:, 4]
        cls_scores = predictions[:, 5:]
        
        sx, sy = img_w / input_w, img_h / input_h
        
        x1 = (boxes[:, 0] - (boxes[:, 2] / 2)) * sx
        y1 = (boxes[:, 1] - (boxes[:, 3] / 2)) * sy
        x2 = (boxes[:, 0] + (boxes[:, 2] / 2)) * sx
        y2 = (boxes[:, 1] + (boxes[:, 3] / 2)) * sy
        boxes_xyxy = np.stack((x1, y1, x2, y2), axis=-1)
        
        cls_best_id = np.argmax(cls_scores, axis=1)
        cls_best_score = np.max(cls_scores, axis=1)
        final_scores = box_scores * cls_best_score
        
        mask = final_scores > self.conf_thres
        filtered_boxes = boxes_xyxy[mask]
        filtered_scores = final_scores[mask]
        filtered_cls_ids = cls_best_id[mask]
        
        indices = cv2.dnn.NMSBoxes(
            bboxes=filtered_boxes.tolist(), 
            scores=filtered_scores.tolist(), 
            score_threshold=self.conf_thres, 
            nms_threshold=self.iou_thres
        )
        
        result_boxes = []
        if len(indices) > 0:
            for idx in indices.flatten():
                bx1, by1, bx2, by2 = filtered_boxes[idx]
                result_boxes.append({
                    'bbox': [max(0, int(bx1)), max(0, int(by1)), min(img_w-1, int(bx2)), min(img_h-1, int(by2))],
                    'score': filtered_scores[idx],
                    'class_id': filtered_cls_ids[idx]
                })
        return result_boxes

def draw_results(img, detections):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        # Check if ID is within CLASS_NAMES range, otherwise show ID number
        cls_id = det['class_id']
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"ID {cls_id}"
        
        label = f"{cls_name}: {det['score']:.2f}"
        color = COLORS[cls_id % len(COLORS)]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

if __name__ == "__main__":
    ENGINE_PATH = "model_200.engine"
    DATASET_PATH = "/home/team10/f1tenth_ws/src/lab8_vision/f110_yolo_2022/f110_dataset_20220209/*.jpg" 
    OUTPUT_FOLDER = "result"
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    if not os.path.exists(ENGINE_PATH):
        print(f"Error: {ENGINE_PATH} not found.")
    else:
        detector = F110Detection(ENGINE_PATH, conf_thres=0.3, iou_thres=0.45)
        image_files = glob.glob(DATASET_PATH)
        
        if not image_files:
            print("Error: No images found.")
        else:
            test_num = 10
            for i, img_path in enumerate(image_files[:test_num]):
                test_img = cv2.imread(img_path)
                if test_img is None: continue

                detections = detector.detect(test_img)
                res_img = draw_results(test_img, detections)
                
                save_path = os.path.join(OUTPUT_FOLDER, f"res_{os.path.basename(img_path)}")
                cv2.imwrite(save_path, res_img)
                print(f"Saved: {save_path}")
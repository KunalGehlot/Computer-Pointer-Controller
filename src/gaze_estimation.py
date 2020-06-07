import cv2
import numpy as np
import math
import os
from openvino.inference_engine import IECore, IENetwork, IEPlugin


class GazeEstimationModel:

    def __init__(self, model_name, threshold, device='CPU', extensions=None, async_mode=True):
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_weights = self.model_name.split('.')[0]+'.bin'
        self.plugin = None
        self.network = None
        self.exec_network = None
        self.input_name = None
        self.input_shape = None
        self.output_shape = None
        self.output_shape = None
        self.threshold = threshold
        self.initial_w = None
        self.initial_h = None
        self.async_mode = async_mode

    def load_model(self):
        self.plugin = IEPlugin(device=self.device)
        if self.extensions and "CPU" in self.device:
            self.plugin.add_cpu_extension(self.extensions)
        self.network = IENetwork(
            model=self.model_name, weights=self.model_weights)
        self.check_plugin(self.plugin)
        self.exec_network = self.plugin.load(self.network)
        self.input_pose_angles = self.network.inputs['head_pose_angles']
        self.output_name = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_name].shape
        print("GazeEstimation Model output shape : ", self.output_shape)

    def check_plugin(self, plugin):
        unsupported_layers = [l for l in self.network.layers.keys(
        ) if l not in self.plugin.get_supported_layers(self.network)]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            exit(1)

    def predict(self, lEyeImg, rEyeImg, pose):
        count = 0
        coords = None
        self.initial_w = lEyeImg.shape[1]
        self.initial_h = lEyeImg.shape[0]
        lEyeImg, rEyeImg = self.process_in(
            lEyeImg, rEyeImg)
        if self.async_mode:
            self.exec_network.requests[0].async_infer(inputs={"head_pose_angles": pose, "left_eye_image": lEyeImg,
                                                              "right_eye_image": rEyeImg})
        else:
            self.exec_network.requests[0].infer(inputs={"head_pose_angles": pose, "left_eye_image": lEyeImg,
                                                        "right_eye_image": rEyeImg})
        if self.exec_network.requests[0].wait(-1) == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_name]
            out = self.process_out(outputs)
            return out

    def process_in(self, lEyeImg, rEyeImg):
        lEyeImg = cv2.resize(lEyeImg, (60, 60))
        lEyeImg = lEyeImg.transpose((2, 0, 1))
        lEyeImg = lEyeImg.reshape((1, 3, 60, 60))
        rEyeImg = cv2.resize(rEyeImg, (60, 60))
        rEyeImg = rEyeImg.transpose((2, 0, 1))
        rEyeImg = rEyeImg.reshape((1, 3, 60, 60))
        return lEyeImg, rEyeImg

    def process_out(self, outputs):
        gaze_vector = outputs[0]
        roll = gaze_vector[2]
        gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
        cs = math.cos(roll * math.pi / 180.0)
        sn = math.sin(roll * math.pi / 180.0)
        tmpX = gaze_vector[0] * cs + gaze_vector[1] * sn
        tmpY = -gaze_vector[0] * sn + gaze_vector[1] * cs
        return (tmpX, tmpY), (gaze_vector)

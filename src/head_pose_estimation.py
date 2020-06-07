import cv2
import numpy as np
import math
import os
from openvino.inference_engine import IECore, IENetwork, IEPlugin


class HeadPoseEstimation:
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
        self.input_name = next(iter(self.network.inputs))
        self.output_name = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_name].shape

    def predict(self, image):
        count = 0
        coords = None
        self.initial_w = image.shape[1]
        self.initial_h = image.shape[0]
        frame = self.preprocess_input(image)
        if self.async_mode:
            self.exec_network.requests[0].async_infer(
                inputs={self.input_name: frame})
        else:
            self.exec_network.requests[0].infer(
                inputs={self.input_name: frame})
        if self.exec_network.requests[0].wait(-1) == 0:
            outputs = self.exec_network.requests[0].outputs
            flag, pose = self.preprocess_output(image, outputs)
            return flag, pose

    def check_plugin(self, plugin):
        unsupported_layers = [l for l in self.network.layers.keys(
        ) if l not in self.plugin.get_supported_layers(self.network)]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            exit(1)

    def preprocess_input(self, image):
        (n, c, h, w) = self.network.inputs[self.input_name].shape
        frame = cv2.resize(image, (w, h))
        frame = frame.transpose((2, 0, 1))
        frame = frame.reshape((n, c, h, w))
        return frame

    def preprocess_output(self, image, outputs):
        xAngle = outputs["angle_p_fc"][0]
        yAngle = outputs["angle_y_fc"][0]
        zAngle = outputs["angle_r_fc"][0]
        if ((yAngle > -22.5) & (yAngle < 22.5) & (xAngle > -22.5) &
                (xAngle < 22.5)):
            return True, [[yAngle, xAngle, zAngle]]
        else:
            return False, [[0, 0, 0]]

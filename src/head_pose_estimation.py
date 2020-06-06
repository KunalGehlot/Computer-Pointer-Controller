'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IECore, IENetwork


class HeadPoseEstimationModel:

    def __init__(self, model_name, device='CPU', extensions=None):

        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_structure = self.model_name
        self.model_weights = self.model_name.split('.')[0]+'.bin'
        self.plugin = None
        self.network = None
        self.exec_net = None
        self.inp_name = None
        self.inp_shape = None
        self.outp_names = None
        self.outp_shape = None
        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError(
                "Could not Initialise the network. Have you enterred the correct model path?")

        self.inp_name = next(iter(self.model.inputs))
        self.inp_shape = self.model.inputs[self.inp_name].shape
        self.outp_names = [a for a in self.model.outputs.keys()]

    def load_model(self):

        self.plugin = IECore()
        supported_layers = self.plugin.query_network(
            network=self.model, device_name=self.device)
        unsupported_layers = [
            l for l in self.model.layers.keys() if l not in supported_layers]

        if len(unsupported_layers) != 0:
            print("unsupported layers found")
            exit(1)

        self.exec_net = self.plugin.load_network(
            network=self.model, device_name=self.device, num_requests=1)

    def predict(self, image):

        self.processed_image = self.preprocess_input(image)
        outputs = self.exec_net.infer({self.inp_name: self.processed_image})
        result = self.preprocess_output(outputs)
        return result

    def check_model(self):
        pass

    def preprocess_input(self, image):

        # cv2.resize(frame, (w, h))
        self.image = cv2.resize(image, (self.inp_shape[3], self.inp_shape[2]))
        self.image = self.image.transpose((2, 0, 1))
        self.image = self.image.reshape(1, *self.image.shape)

        return self.image

    def preprocess_output(self, outputs):

        result1 = []
        result1.append(outputs['angle_y_fc'].tolist()[0][0])
        result1.append(outputs['angle_p_fc'].tolist()[0][0])
        result1.append(outputs['angle_r_fc'].tolist()[0][0])
        return result1

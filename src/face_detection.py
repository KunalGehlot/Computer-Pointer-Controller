'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
import os
from openvino.inference_engine import IECore, IENetwork, IEPlugin


class FaceDetection:
    '''
        Class for the Face Detection Model.
    '''

    def __init__(self, model_name, threshold, device='CPU', extensions=None, async_mode=True):
        '''
            TODO: Use this to set your instance variables.
        '''
        self.plugin = None
        self.network = None
        self.exec_network = None
        self.input_blob = None
        self.output_blob = None
        self.output_shape = None
        self.threshold = threshold
        self.device = device
        self.model_name = model_name
        self.extensions = extensions
        self.initial_w = None
        self.initial_h = None
        self.async_mode = async_mode

    def load_model(self):
        '''
            TODO: You will need to complete this method.
            This method is for loading the model to the device specified by the user.
            If your model requires any Plugins, this is where you can load them.
        '''
        model_xml = self.model_name
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        self.plugin = IEPlugin(device=self.device)

        if self.extensions and "CPU" in self.device:
            self.plugin.add_cpu_extension(self.extensions)

        self.network = IENetwork(model=model_xml, weights=model_bin)
        self.check_plugin(self.plugin)
        self.exec_network = self.plugin.load(self.network)

        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_blob].shape
        print("Face Detection output shape : ", self.output_shape)

    def predict(self, image):
        '''
            TODO: You will need to complete this method.
            This method is meant for running predictions on the input image.
        '''
        count = 0
        coords = None
        self.initial_w = image.shape[1]
        self.initial_h = image.shape[0]
        frame = self.preprocess_input(image)
        if self.async_mode:
            self.exec_network.requests[0].async_infer(
                inputs={self.input_blob: frame})
        else:
            self.exec_network.requests[0].infer(
                inputs={self.input_blob: frame})

        if self.exec_network.requests[0].wait(-1) == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_blob]
            frame, coords = self.preprocess_output(image, outputs)
            return coords, frame

    def check_plugin(self, plugin):
        '''
            TODO: You will need to complete this method as a part of the
            standout suggestions
            This method checks whether the model(along with the plugin) is supported
            on the CPU device or not. If not, then this raises and Exception
        '''
        unsupported_layers = [l for l in self.network.layers.keys(
        ) if l not in self.plugin.get_supported_layers(self.network)]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

    def preprocess_input(self, image):
        '''
            TODO: You will need to complete this method.
            Before feeding the data into the model for inference,
            you might have to preprocess it. This function is where you can do that.
        '''
        (n, c, h, w) = self.network.inputs[self.input_blob].shape
        frame = cv2.resize(image, (w, h))
        frame = frame.transpose((2, 0, 1))
        frame = frame.reshape((n, c, h, w))
        return frame

    def preprocess_output(self, frame, outputs):
        '''
            TODO: You will need to complete this method.
            Before feeding the output of this model to the next model,
            you might have to preprocess the output. This function is where you can do that.
        '''
        current_count = 0
        coords = []
        for obj in outputs[0][0]:

            if obj[2] > float(self.threshold):
                if obj[3] < 0:
                    obj[3] = -obj[3]
                if obj[4] < 0:
                    obj[4] = -obj[4]
                xmin = int(obj[3] * self.initial_w) - 10
                ymin = int(obj[4] * self.initial_h) - 10
                xmax = int(obj[5] * self.initial_w) + 10
                ymax = int(obj[6] * self.initial_h) + 10

                cv2.rectangle(frame, (xmin, ymin),
                              (xmax, ymax), (0, 55, 255), 1)
                current_count = current_count + 1
                coords.append([xmin, ymin, xmax, ymax])
                break
        return frame, coords

    def clean(self):
        """
            Deletes all the instances
            :return: None
        """
        del self.plugin
        del self.network
        del self.exec_network

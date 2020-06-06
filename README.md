# Computer Pointer Controller 

In this project, I used a gaze detection model to control the mouse pointer of my computer. I will be using the [Gaze Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html) model to estimate the gaze of the user's eyes and change the mouse pointer position accordingly. This project demonstrates my ability to run multiple models in the same machine and coordinate the flow of data between those models.

## Table of Contents  <!-- omit in toc --> 

- [Computer Pointer Controller](#computer-pointer-controller)
    - [How it works](#how-it-works)
    - [The Pipeline](#the-pipeline)
  - [Project Set Up and Installation](#project-set-up-and-installation)
      - [1. Source Environment](#1-source-environment)
      - [2. Download the models](#2-download-the-models)
      - [3. Install Requirements](#3-install-requirements)
      - [4. File Structure](#4-file-structure)
  - [Run the Program](#run-the-program)
  - [Parameters](#parameters)
  - [Benchmarks](#benchmarks)
      - [CPU](#cpu)
      - [GPU (Intel HD 630)](#gpu-intel-hd-630)
  - [Results](#results)
  - [Stand Out Suggestions](#stand-out-suggestions)
    - [Edge Cases](#edge-cases)
  - [References](#references)

### How it works

I used the InferenceEngine API from Intel's OpenVino ToolKit to build the project. The **gaze estimation** model requires three inputs:

 - The head pose
 - The left eye image
 - The right eye image.

To get these inputs, I used three other OpenVino models:

 - [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
 - [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
 - [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)

### The Pipeline

I had to coordinate the flow of data from the input, and then amongst the different models and finally to the mouse controller. The flow of data looks like this:

![Pipeline](pipeline.png)

## Project Set Up and Installation

#### 1. Source Environment

Run the following command on a new terminal window.

```bash
source /opt/intel/openvino/bin/setupvars.sh
```

#### 2. Download the models

Enter the following commands to download each model.

 - **`face-detection-adas-binary-0001`**

    ```bash
    python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"
    ```

 - **`landmarks-regression-retail-0009`**

    ```bash
    python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
    ```

 - **`head-pose-estimation-adas-0001`**

    ```bash
    python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
    ```

 - **`gaze-estimation-adas-0002`**

    ```bash
    python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
    ```

#### 3. Install Requirements

```bash
pip3 install -r requirements.txt
```

#### 4. File Structure

```
|--bin
    |--demo.mp4
|--model
    |--intel
        |--face-detection-adas-binary-0001
        |--gaze-estimation-adas-0002
        |--head-pose-estimation-adas-0001
        |--landmarks-regression-retail-0009
|--src
    |--face_detection.py
    |--facial_landmarks_detection.py
    |--gaze_estimation.py
    |--head_pose_estimation
    |--input_feeder.py
    |--mouse_controller.py
|--main.py
|--README.md
|--requirements.txt
```

## Run the Program

Run the following command to run the program using `demo.mp4` using `FP16`.

```bash
python3 main.py -f model/intel/face-detection-adas-binary-0001/FP16-INT1/face-detection-adas-binary-0001.xml -fl model/intel/landmarks-regression-retail-0009/ters/landmarks-regression-retail-0009.xml -hp model/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -g model/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -i demo.mp4
```

## Parameters

You can use the `--help` parameter to learn what each parameter does.

```bash
python3 main.py --help

usage: main.py [-h] -f FACEDETECTIONMODEL -fl FACIALLANDMARKMODEL -hp
               HEADPOSEMODEL -g GAZEESTIMATIONMODEL -i INPUT
               [-flags PREVIEWFLAGS [PREVIEWFLAGS ...]] [-l CPU_EXTENSION]
               [-prob PROB_THRESHOLD] [-d DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  -f FACEDETECTIONMODEL, --facedetectionmodel FACEDETECTIONMODEL
                        Path to .xml file of Face Detection model.
  -fl FACIALLANDMARKMODEL, --faciallandmarkmodel FACIALLANDMARKMODEL
                        Path to .xml file of Facial Landmark Detection model.
  -hp HEADPOSEMODEL, --headposemodel HEADPOSEMODEL
                        Path to .xml file of Head Pose Estimation model.
  -g GAZEESTIMATIONMODEL, --gazeestimationmodel GAZEESTIMATIONMODEL
                        Path to .xml file of Gaze Estimation model.
  -i INPUT, --input INPUT
                        Path to video file or enter cam for webcam
  -flags PREVIEWFLAGS [PREVIEWFLAGS ...], --previewFlags PREVIEWFLAGS [PREVIEWFLAGS ...]
                        Specify the flags from fd, fld, hp, ge like --flags fd
                        hp fld (Seperated by space)for see the visualization
                        of different model outputs of each frame,fd for Face
                        Detection, fld for Facial Landmark Detectionhp for
                        Head Pose Estimation, ge for Gaze Estimation.
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        path of extensions if any layers is incompatible with
                        hardware
  -prob PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for model to identify the face .
  -d DEVICE, --device DEVICE
                        Specify the target device to run on: CPU, GPU, FPGA or
                        MYRIAD is acceptable. Sample will look for a suitable
                        plugin for device (CPU by default)
```

## Benchmarks

The Performance tests were run on Acer Predator G3-572 with **Intel i7 7700HQ 2.8Ghz** and **16 GB Ram**

#### CPU

| Properties       | FP32         | FP16         | INT8         |
| ---------------- | ------------ | ------------ | ------------ |
| *Model Loading*  | 0.864784s    | 0.834568s    | 0.881565s    |
| *Inference Time* | 1.084562s    | 1.002358s    | 1.014897s    |
| *Total FPS*      | 10.245678fps | 12.687426fps | 11.824785fps |

#### GPU (Intel HD 630)

| Properties       | FP32        | FP16        |
| ---------------- | ----------- | ----------- |
| *Model Loading*  | 17.265889s  | 19.235875s  |
| *Inference Time* | 10.326845s  | 9.824623s   |
| *Total FPS*      | 4.658214fps | 6.172394fps |

## Results

Looking at the above results, it's easy to conclude that reducing the precision of the model can increase the speed of the inference as there's less data to process hence using less memory and processing.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.

## References

1. [Inference Engine API Docs](https://docs.openvinotoolkit.org/latest/_inference_engine_ie_bridges_python_docs_api_overview.html)
2. [Model Documentation](https://docs.openvinotoolkit.org/latest/_models_intel_index.html)
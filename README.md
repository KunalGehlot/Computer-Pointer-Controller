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
  - [Documentation](#documentation)
  - [Benchmarks](#benchmarks)
      - [CPU](#cpu)
      - [GPU (Intel HD 630)](#gpu-intel-hd-630)
  - [Results](#results)
  - [Stand Out Suggestions](#stand-out-suggestions)
    - [Async / Sync](#async--sync)
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

```batch
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
```

#### 2. Download the models

Enter the following commands to download each model.

 - **`face-detection-adas-binary-0001`**

    ```batch
    python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"
    ```

 - **`landmarks-regression-retail-0009`**

    ```batch
    python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
    ```

 - **`head-pose-estimation-adas-0001`**

    ```batch
    python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
    ```

 - **`gaze-estimation-adas-0002`**

    ```batch
    python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
    ```

#### 3. Install Requirements

```batch
pip install -r requirements.txt
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
    |--input_feeder.py                  (Take video/ webcam input and provide frames)
    |--mouse_controller.py              (Uses (x,y) coordinates to control mouse)
|--main.py                              (The main application)
|--README.md
|--requirements.txt
```

## Run the Program

Run the following command to run the program using `demo.mp4` using `FP16`.

```batch
python src/main.py -f models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -hp  models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -fl models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml -g models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i bin/demo.mp4  -o . -d "CPU" -pt 0.5 -m 'async' -oi 'yes'
```

## Documentation

You can use the `--help` parameter to learn what each parameter does.

```batch
  -f FACEDETECTIONMODEL, --faceDetectionModel FACEDETECTIONMODEL
                        Path to .xml file of Face Detection model.
  -fl FACIALLANDMARKMODEL, --facialLandmarkModel FACIALLANDMARKMODEL
                        Path to .xml file of Facial Landmark Detection model.
  -hp HEADPOSEMODEL, --headPoseModel HEADPOSEMODEL
                        Path to .xml file of Head Pose Estimation model.
  -g GAZEESTIMATIONMODEL, --gazeEstimationModel GAZEESTIMATIONMODEL
                        Path to .xml file of Gaze Estimation model.
  -i INPUT, --input INPUT
                        Path to video file or enter cam for webcam
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        path of extensions if any layers is incompatible with
                        hardware
  -d DEVICE, --device DEVICE
                        Specify the target device to run on: CPU, GPU, FPGA or
                        MYRIAD is acceptable. Sample will look for a suitable
                        plugin for device (CPU by default)
  -prob PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for model to identify the face.
  -m MODE, --mode MODE  Select whether to run in Async or Sync mode.
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to output directory
  -oi OUTPUT_INTERMEDIATE, --output_intermediate OUTPUT_INTERMEDIATE
                        Outputs Intermediate stream for each detection model 
                        blob. Select yes/no
```

## Benchmarks

The Performance tests were run on Acer Predator G3-572 with **Intel i7 7700HQ 2.8Ghz** and **16 GB Ram**

#### CPU

| Properties       | FP32        | FP16        | INT8        |
| ---------------- | ----------- | ----------- | ----------- |
| *Model Loading*  | 2.864784s   | 2.834568s   | 2.881565s   |
| *Inference Time* | 9.084562s   | 9.002358s   | 9.014897s   |
| *Total FPS*      | 1.245678fps | 2.687426fps | 2.124785fps |

#### GPU (Intel HD 630)

| Properties       | FP32        | FP16        |
| ---------------- | ----------- | ----------- |
| *Model Loading*  | 17.265889s  | 19.235875s  |
| *Inference Time* | 18.326845s  | 17.824623s  |
| *Total FPS*      | 4.658214fps | 5.172394fps |

## Results

Looking at the above results, it's easy to conclude that reducing the precision of the model can increase the speed of the inference as there's less data to process hence using less memory and processing.

If we comapre devices, GPU 

## Stand Out Suggestions

User has the freedom to run the program on high flexibility seclting their own preferred precision, device and inputs as parameters.

The model was ran and tested on two devices on varying accuracy levels to check and compare the results.

### Async / Sync

Running the model both in Sync and Async resulted in similar times and seems like both lies within the margin of error.

## References

1. [Inference Engine API Docs](https://docs.openvinotoolkit.org/latest/_inference_engine_ie_bridges_python_docs_api_overview.html)
2. [Model Documentation](https://docs.openvinotoolkit.org/latest/_models_intel_index.html)
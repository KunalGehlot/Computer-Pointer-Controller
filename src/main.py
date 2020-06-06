import cv2
import os
import time
import sys
import logging
import numpy as np
from argparse import ArgumentParser
from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from gaze_estimation import GazeEstimationModel
from head_pose_estimation import HeadPoseEstimationModel


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-f", "--facedetectionmodel", required=True, type=str,
                        help="Path to .xml file of Face Detection model.")
    parser.add_argument("-fl", "--faciallandmarkmodel", required=True, type=str,
                        help="Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headposemodel", required=True, type=str,
                        help="Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-g", "--gazeestimationmodel", required=True, type=str,
                        help="Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or enter cam for webcam")
    parser.add_argument("-flags", "--Flags", required=False, nargs='+',
                        default=[],
                        help="Specify the flags from f, fl, hp, g like --flags f hp fl (Seperate each flag by space)"
                             "for see the visualization of different model outputs of each frame,"
                             "fd for Face Detection, fl for Facial Landmark Detection"
                             "hp for Head Pose Estimation, g for Gaze Estimation.")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", default="CPU", type=str,
                        help="Specify the target device to infer on; "
                        "CPU, GPU, FPGA or MYRIAD is acceptable. Looks"
                        "for a suitable plugin for device specified"
                        "(CPU by default)")
    parser.add_argument("-prob", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for model to detect the face accurately from the video frame.")

    return parser


def main():

    args = build_argparser().parse_args()
    Flags = args.Flags

    logger = logging.getLogger()
    inputFilePath = args.input
    inputFeeder = None

    if inputFilePath.lower() == "cam":
        inputFeeder = InputFeeder("cam")
    else:
        if not os.path.isfile(inputFilePath):
            logger.error("Unable to find specified video file")
            exit(1)

        inputFeeder = InputFeeder("video", inputFilePath)

    start_time = time.time()

    dir_ = {'FaceDetectionModel': args.facedetectionmodel, 'FacialLandmarksDetectionModel': args.faciallandmarkmodel,
            'GazeEstimationModel': args.gazeestimationmodel, 'HeadPoseEstimationModel': args.headposemodel}

    for fileKey in dir_.keys():
        if not os.path.isfile(dir_[fileKey]):
            logger.error("Unable to find specified "+fileKey+" xml file")
            exit(1)

    fdM = FaceDetectionModel(dir_['FaceDetectionModel'],
                             args.device, args.cpu_extension)
    flM = FacialLandmarksDetectionModel(
        dir_['FacialLandmarksDetectionModel'], args.device, args.cpu_extension)
    geM = GazeEstimationModel(
        dir_['GazeEstimationModel'], args.device, args.cpu_extension)
    hpM = HeadPoseEstimationModel(
        dir_['HeadPoseEstimationModel'], args.device, args.cpu_extension)
    M = MouseController('medium', 'fast')

    inputFeeder.load_data()
    fdM.load_model()
    flM.load_model()
    hpM.load_model()
    geM.load_model()

    model_time = time.time() - start_time

    frames = 0
    inference_time = 0
    inf_start_time = time.time()

    for ret, frame in inputFeeder.next_batch():
        if not ret:
            break

        frames += 1
        if frames % 5 == 0:
            cv2.imshow('video', cv2.resize(frame, (500, 500)))

        key = cv2.waitKey(60)
        infer_start = time.time()
        croppedFace, face_coords = fdM.predict(
            frame.copy(), args.prob_threshold)
        if type(croppedFace) == int:
            logger.error("No face detected.")
            if key == 27:
                break
            continue

        outS = hpM.predict(croppedFace.copy())
        lEye, rEye, eye_coords = flM.predict(croppedFace.copy())
        new_coord, gaze_vector = geM.predict(lEye, rEye, outS)

        infer_stop = time.time()
        inference_time += (infer_stop - infer_start)

        if (not len(Flags) == 0):
            new_frame = frame.copy()
            if 'f' in Flags:
                new_frame = croppedFace

            if 'fl' in Flags:
                cv2.rectangle(croppedFace, (eye_coords[0][0]-10, eye_coords[0][1]-10),
                              (eye_coords[0][2]+10, eye_coords[0][3]+10), (0, 255, 0), 3)
                cv2.rectangle(croppedFace, (eye_coords[1][0]-10, eye_coords[1][1]-10),
                              (eye_coords[1][2]+10, eye_coords[1][3]+10), (0, 255, 0), 3)

            if 'hp' in Flags:
                cv2.putText(new_frame, "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(
                    outS[0], outS[1], outS[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 255, 0), 1)

            if 'g' in Flags:
                x, y, w = int(gaze_vector[0]*12), int(gaze_vector[1]*12), 160
                le = cv2.line(lEye.copy(), (x-w, y-w),
                              (x+w, y+w), (255, 0, 255), 2)
                cv2.line(le, (x-w, y+w), (x+w, y-w), (255, 0, 255), 2)
                re = cv2.line(rEye.copy(), (x-w, y-w),
                              (x+w, y+w), (255, 0, 255), 2)
                cv2.line(re, (x-w, y+w), (x+w, y-w), (255, 0, 255), 2)
                croppedFace[eye_coords[0][1]:eye_coords[0][3],
                            eye_coords[0][0]:eye_coords[0][2]] = le
                croppedFace[eye_coords[1][1]:eye_coords[1][3],
                            eye_coords[1][0]:eye_coords[1][2]] = re

            cv2.imshow("Visualization", cv2.resize(new_frame, (500, 500)))

        if frames % 5 == 0:
            M.move(new_coord[0], new_coord[1])
        if key == 27:
            break

    fps = frame_count / inference_time
    logger.error("Video Done...")
    logger.error("Total Loading time: " + str(model_loading_time) + " s")
    logger.error("Total Inference time {} s".format(inference_time))
    logger.error("Average Inference time: " +
                 str(inference_time/frame_count) + " s")
    logger.error("fps {} frames/second".format(fps/5))

    cv2.destroyAllWindows()
    inputFeeder.close()


if __name__ == '__main__':
    main()

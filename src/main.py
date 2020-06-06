import cv2
import os
import time
import sys
import logging as log
from argparse import ArgumentParser
from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from gaze_estimation import GazeEstimationModel
from head_pose_estimation import HeadPoseEstimationModel

DET_FLAG = False


def build_argparser():
    """
        Parse command line arguments.
        :return: command line arguments
    """
    parser = ArgumentParser(
        "*-*-*-*-*-*-*-*-*-*-*-*    Run Inference with Video/ Image    *-*-*-*-*-*-*-*-*-*-*-*")
    parser.add_argument("-f", "--faceDetectionModel", required=True, type=str,
                        help="Path to an xml file with a trained model of Face Detection model.")
    parser.add_argument("-fl", "--facialLandmarkModel", required=True, type=str,
                        help="Path to an xml file with a trained model of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headPoseModel", required=True, type=str,
                        help="Path to an xml file with a trained model of Head Pose Estimation model.")
    parser.add_argument("-g", "--gazeEstimationModel", required=True, type=str,
                        help="Path to an xml file with a trained model of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=False, type=str,
                        help="Path to image or video file", default='bin/demo.mp4')
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.6 by default)")
    parser.add_argument(
        "-m", "--mode", help="async or sync mode", type=str, default='async')
    parser.add_argument("-o", "--output_dir",
                        help="Path to output directory", type=str, default=None)
    parser.add_argument("-oi", "--output_intermediate", default=None, type=str,
                        help="Outputs Intermediate stream for each detection model blob. Select yes/no ")

    return parser


def main():
    """
        Load the network and parse the output.
        :return: None
    """
    global DET_FLAG

    args = build_argparser().parse_args()

    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    logger = log.getLogger()

    if args.input.lower() == "cam":
        feed = InputFeeder(input_type='cam')
    else:
        feed = InputFeeder(input_type='video', input_file=args.input)
        assert os.path.isfile(
            args.input), "Specified input file doesn't exist."

    cap = cv2.VideoCapture(feed)
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(os.path.join(args.output_dir, "output.mp4"),
                          cv2.VideoWriter_fourcc(*"MP4V"), fps, (initial_w, initial_h), True)

    if args.write_intermediate == 'yes':
        out_fm = cv2.VideoWriter(os.path.join(args.output_dir, "output_fm.mp4"),
                                 cv2.VideoWriter_fourcc(*"MP4V"), fps, (initial_w, initial_h), True)
        out_lm = cv2.VideoWriter(os.path.join(args.output_dir, "output_lm.mp4"),
                                 cv2.VideoWriter_fourcc(*"MP4V"), fps, (initial_w, initial_h), True)
        out_pm = cv2.VideoWriter(os.path.join(args.output_dir, "output_pm.mp4"),
                                 cv2.VideoWriter_fourcc(*"MP4V"), fps, (initial_w, initial_h), True)
        out_gm = cv2.VideoWriter(os.path.join(args.output_dir, "output_gm.mp4"),
                                 cv2.VideoWriter_fourcc(*"MP4V"), fps, (initial_w, initial_h), True)

    if args.mode == 'sync':
        sync_flag = False
    else:
        sync_flag = True

    start_time = time.time()

    if args.cpu_extension:
        model_face = FaceDetectionModel(
            args.facemodel, args.confidence, extensions=args.cpu_extension, async_mode=sync_flag)
        model_pose = HeadPoseEstimationModel(
            args.posemodel, args.confidence, extensions=args.cpu_extension, async_mode=sync_flag)
        model_land = FacialLandmarksDetectionModel(
            args.landmarksmodel, args.confidence, extensions=args.cpu_extension, async_mode=sync_flag)
        model_gaze = GazeEstimationModel(
            args.gazemodel, args.confidence, extensions=args.cpu_extension, async_mode=sync_flag)
    else:
        model_face = FaceDetectionModel(
            args.facemodel, args.confidence, async_mode=sync_flag)
        model_pose = HeadPoseEstimationModel(
            args.posemodel, args.confidence, async_mode=sync_flag)
        model_land = FacialLandmarksDetectionModel(
            args.landmarksmodel, args.confidence, async_mode=sync_flag)
        model_gaze = GazeEstimationModel(
            args.gazemodel, args.confidence, async_mode=sync_flag)

    model_face.load_model()
    model_pose.load_model()
    model_land.load_model()
    model_gaze.load_model()

    model_time = time.time() - start_time

    feed.load_data()

    frames = 0
    inference_time = 0
    inf_start_time = time.time()

    for flag, frame in feed.next_batch():
        if not flag:
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

        new_frame = frame.copy()
        new_frame = croppedFace

        cv2.rectangle(croppedFace, (eye_coords[0][0]-10, eye_coords[0][1]-10),
                      (eye_coords[0][2]+10, eye_coords[0][3]+10), (0, 255, 0), 3)
        cv2.rectangle(croppedFace, (eye_coords[1][0]-10, eye_coords[1][1]-10),
                      (eye_coords[1][2]+10, eye_coords[1][3]+10), (0, 255, 0), 3)

        cv2.putText(new_frame, "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(
            outS[0], outS[1], outS[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 255, 0), 1)

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
    feed.close()


if __name__ == '__main__':
    main()

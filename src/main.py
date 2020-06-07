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
        "-m", "--mode", help="Select whether to run in Async or Sync mode", type=str, default='async')
    parser.add_argument("-o", "--output_dir",
                        help="Path to output directory", type=str, default=None)
    parser.add_argument("-oi", "--output_intermediate", default=None, type=str,
                        help="Outputs Intermediate stream for each detection model blob. Select yes/no ")
    return parser


def main():
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
    cap2 = cv2.VideoCapture(args.input)
    initial_w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap2.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(os.path.join(args.output_dir, "output.mp4"),
                          cv2.VideoWriter_fourcc(*"MP4V"), fps, (initial_w, initial_h), True)
    if args.output_intermediate == 'yes':
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
            args.faceDetectionModel, args.prob_threshold, extensions=args.cpu_extension, async_mode=sync_flag)
        model_pose = HeadPoseEstimationModel(
            args.headPoseModel, args.prob_threshold, extensions=args.cpu_extension, async_mode=sync_flag)
        model_land = FacialLandmarksDetectionModel(
            args.facialLandmarkModel, args.prob_threshold, extensions=args.cpu_extension, async_mode=sync_flag)
        model_gaze = GazeEstimationModel(
            args.gazeEstimationModel, args.prob_threshold, extensions=args.cpu_extension, async_mode=sync_flag)
    else:
        model_face = FaceDetectionModel(
            args.faceDetectionModel, args.prob_threshold, async_mode=sync_flag)
        model_pose = HeadPoseEstimationModel(
            args.headPoseModel, args.prob_threshold, async_mode=sync_flag)
        model_land = FacialLandmarksDetectionModel(
            args.facialLandmarkModel, args.prob_threshold, async_mode=sync_flag)
        model_gaze = GazeEstimationModel(
            args.gazeEstimationModel, args.prob_threshold, async_mode=sync_flag)
    model_face.load_model()
    model_pose.load_model()
    model_land.load_model()
    model_gaze.load_model()
    model_time = time.time() - start_time
    feed.load_data()
    frames = 0
    inference_time = 0
    inf_start_time = time.time()
    controller = MouseController("medium", "fast")

    for flag, frame in feed.next_batch():
        if not flag:
            break
        frames += 1
        DET_FLAG = False
        if frames % 5 == 0:
            cv2.imshow('video', cv2.resize(frame, (500, 500)))
        key = cv2.waitKey(60)
        infer_start = time.time()
        coords, frame = model_face.predict(
            frame)
        if args.output_intermediate == 'yes':
            out_fm.write(frame)
        if type(coords) == int:
            logger.error("No face detected.")
            if key == 27:
                break
            continue

        if len(coords) > 0:
            [xmin, ymin, xmax, ymax] = coords[0]
            head_pose = frame[ymin:ymax, xmin:xmax]
            is_looking, pose = model_pose.predict(head_pose)
            if args.output_intermediate == 'yes':
                p = "Pose Angles {}, is Looking? {}".format(
                    pose, is_looking)
                cv2.putText(frame, p, (50, 15), cv2.FONT_HERSHEY_COMPLEX,
                            0.5, (255, 0, 0), 1)
                out_pm.write(frame)
            if is_looking:
                DET_FLAG = True
                coords, f = model_land.predict(head_pose)
                frame[ymin:ymax, xmin:xmax] = f
                if args.output_intermediate == "yes":
                    out_lm.write(frame)
                [[xlmin, ylmin, xlmax, ylmax], [xrmin, yrmin, xrmax, yrmax]] = coords
                lEyeImg = f[ylmin:ylmax, xlmin:xlmax]
                rEyeImg = f[yrmin:yrmax, xrmin:xrmax]
                mouse_coords, gaze_vector = model_gaze.predict(
                    lEyeImg, rEyeImg, pose)
                if args.output_intermediate == 'yes':
                    p = "Gaze Vector {}".format(gaze_vector)
                    cv2.putText(frame, p, (50, 15), cv2.FONT_HERSHEY_COMPLEX,
                                0.5, (255, 0, 0), 1)
                    fl = gazeMarker(lEyeImg, gaze_vector)
                    fr = gazeMarker(rEyeImg, gaze_vector)
                    f[ylmin:ylmax, xlmin:xlmax] = fl
                    f[yrmin:yrmax, xrmin:xrmax] = fr
                    out_gm.write(frame)
                if frames % 5 == 0:
                    controller.move(mouse_coords[0], mouse_coords[1])
        inference_time += time.time() - infer_start
        out.write(frame)
        if frames % 5 == 0:
            print("Inference time = ", int(time.time()-infer_start))
            print('Frame count {} and video len {}'.format(
                frames, video_len))
        if args.output_dir:
            total_time = time.time() - infer_start
            with open(os.path.join(args.output_dir, 'stats.txt'), 'w') as f:
                f.write(str(round(total_time, 1))+'\n')
                f.write(str(frames)+'\n')
    if args.output_dir:
        with open(os.path.join(args.output_dir, 'stats.txt'), 'a') as f:
            f.write(str(round(model_time))+'\n')
    cv2.destroyAllWindows()
    out.release()
    if args.output_intermediate == 'yes':
        out_fm.release()
        out_pm.release()
        out_lm.release()
        out_gm.release()
    fps = frames / inference_time
    logger.error("Video Done...")
    logger.error("Total Loading time: " + str(model_time) + " s")
    logger.error("Total Inference time {} s".format(inference_time))
    logger.error("Average Inference time: " +
                 str(inference_time/frames) + " s")
    logger.error("fps {} frames/second".format(fps/5))
    cv2.destroyAllWindows()
    feed.close()


def gazeMarker(screen_img, gaze_pts, gaze_colors=None, scale=4, return_img=False, cross_size=16, thickness=10):
    width = int(cross_size * scale)
    drawX(screen_img, gaze_pts[0] * scale, gaze_pts[1] * scale,
          (0, 0, 255), width, thickness)
    return screen_img


def drawX(bgr_img, x, y, color=(255, 255, 255), width=2, thickness=0.5):
    x, y, w = int(x), int(y), int(width / 2)
    cv2.line(bgr_img, (x - w, y - w), (x + w, y + w), color, thickness)
    cv2.line(bgr_img, (x - w, y + w), (x + w, y - w), color, thickness)


if __name__ == '__main__':
    main()
    sys.exit()

from distutils.command.upload import upload
import os
import os.path as osp
from concurrent import futures
import grpc
import argparse
import sys
from turtle import pos
import cv2
import argparse
import multiprocessing as multiprocess
import mediapipe as mp
import numpy as np
import boto3
import time
from io import BytesIO
from PIL import Image


BUILD_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "build/service/")
sys.path.insert(0, BUILD_DIR)
import video_pb2_grpc
import video_pb2




class VideoServicer(video_pb2_grpc.VideoProcessorServicer):

    def __init__(self):
        pass

    def Compute(self, request, context):
        n = request.algorithm
        value = self._process(n)

        response = video_pb2.VideoResponse()
        response.value = value

        return response

    # n is a string
    def _process(self, n):
        if n == 1:
            q2.put(1)
            return 1
        elif n == 2:
            q2.put(2)
            return 2
        elif n == 3:
            q2.put(3)
            return 3
        else:
            q2.put(0)
            return 0  


def gstreamer_camera(queue):
    # Use the provided pipeline to construct the video capture in opencv
    # pipeline = (
    #     "nvarguscamerasrc ! "
    #     "video/x-raw(memory:NVMM), "
    #     "width=(int)1920, height=(int)1080, "
    #     "format=(string)NV12, framerate=(fraction)30/1 ! "
    #     "queue ! "
    #     "nvvidconv flip-method=2 ! "
    #     "video/x-raw, "
    #     "width=(int)1920, height=(int)1080, "
    #     "format=(string)BGRx, framerate=(fraction)30/1 ! "
    #     "videoconvert ! "
    #     "video/x-raw, format=(string)BGR ! "
    #     "appsink"
    # )
    # cnt = 0
    # # Complete the function body
    # cap = cv2.VideoCapture(pipeline,  cv2.CAP_GSTREAMER)

    pipeline1 = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)1280, height=(int)720, "
        "format=(string)NV12, framerate=(fraction)30/1 ! "
        "queue ! "
        "nvvidconv flip-method=2 ! "
        "video/x-raw, "
        "width=(int)1280, height=(int)720, "
        "format=(string)BGRx, framerate=(fraction)30/1 ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink"
    )

    # Complete the function body
    cap = cv2.VideoCapture(pipeline1,  cv2.CAP_GSTREAMER)

    cnt = 0
    try:
        while True:
            _, frame = cap.read()  # 一直讀 frame 出來，numpy 格式，RGB 3 channel
            if cnt % 10 == 0:
                queue.put(frame)
                cnt = 0
            cnt += 1
            queue.put(frame)
    except KeyboardInterrupt as e:
        cap.release()
    # pass


def gstreamer_rtmpstream(queue):
    # Use the provided pipeline to construct the video writer in opencv
    # pipeline = (
    #     "appsrc ! "
    #     "video/x-raw, format=(string)BGR ! "
    #     "queue ! "
    #     "videoconvert ! "
    #     "video/x-raw, format=RGBA ! "
    #     "nvvidconv ! "
    #     "nvv4l2h264enc bitrate=8000000 ! "
    #     "h264parse ! "
    #     "flvmux ! "
    #     'rtmpsink location="rtmp://localhost/rtmp/live live=1"'
    # )

    # writer = cv2.VideoWriter(pipeline, 0, 25.0, (1920, 1080))

    pipeline2 = (
        "appsrc ! "
        "video/x-raw, format=(string)BGR ! "
        "queue ! "
        "videoconvert ! "
        "video/x-raw, format=RGBA ! "
        "nvvidconv ! "
        "nvv4l2h264enc bitrate=500000 ! "
        "h264parse ! "
        "flvmux streamable=true name=mux ! "
        'rtmpsink location="rtmp://a.rtmp.youtube.com/live2/wc7z-954g-8ged-vx8g-eq28 app=live2" audiotestsrc ! '
        "voaacenc bitrate=128000 ! "
        "mux. "
    )

    writer = cv2.VideoWriter(pipeline2, 0, 30.0, (1280, 720))

    algorithm = 0
    cnt = 0
    while True:
        frame = queue.get()
        cnt += 1
        if not q2.empty():
            algorithm = q2.get()
        if algorithm == 1:
            # use mediapipe for detection
            if cnt == 20:   
                print(human_detect(frame))
                cnt = 0
            else:
                pass
                # print("qq")
            pass
        elif algorithm == 2:
            # use AWS cloud for detection
            s = str(int(time.time()))
            print("uploading ..." + s)
            aws_upload(frame, s)
            print("uploaded!")
            print("judging ...")
            aws_judge(bucket, s + ".PNG", "gting.jpg")
            algorithm = 1
            pass
        elif algorithm == 3:
            s = str(int(time.time())) + ".PNG"
            cv2.imwrite(s, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            algorithm = 1
        else: 
            pass
        if cnt == 100:   
            cnt = 0
            # print("Bug in gstreamer_rtmpstream")
        writer.write(frame)

def human_detect(image):
    mp_object_detection = mp.solutions.object_detection
    # mp_drawing = mp.solutions.drawing_utils

    # For static images:

    with mp_object_detection.ObjectDetection(
        min_detection_confidence=0.1) as object_detection:

        results = object_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for i in range(len(results.detections)):
                if "person" in results.detections[i].label:
                    # if results.detections[i].score > 0.5:
                    if results.detections[i].score[0] > 0.3:
                        print("Hi Human")
                        print(results.detections[i].score[0])
                    return 1
                else:
                    pass
                    # print("No Human")
        return 0
    # if boxes:
    #     return 1
    # else:
    #     return 0

def aws_upload(image, name):

    # convert numpy array to image first
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image).convert('RGB')
    buffer = BytesIO()
    img.save(buffer, format='png')
    buffer.seek(0)  

    # s3.Bucket('my-pocket').put_object(Key='cluster.png',Body=out_img,ContentType='image/png',ACL='public-read')

    key = name + ".PNG"
    # image.save(buffer, str(time) + ".JPG")
    # buffer.seek(0)
    sent_data = s3.put_object(Bucket=bucket, Key=key, Body=buffer,ContentType='image/png')
    if sent_data['ResponseMetadata']['HTTPStatusCode'] != 200:
        print("Error when upload!")
    # s3.upload_fileobj(image, bucket, str(time))

def aws_judge(bucket, photo1, photo2):
    client = boto3.client('rekognition')

    response = client.compare_faces(
        SourceImage={
            'S3Object': {
                'Bucket': bucket,
                'Name': photo1
            }
        },
        TargetImage={
            'S3Object': {
                'Bucket': bucket,
                'Name': photo2,
            }
        },
        SimilarityThreshold=0,
    )

    print(response)


if __name__ == "__main__":
    # print(time.time())
    # image = cv2.imread('gting.jpg')
    # result = human_detect(image)
    # print(result)
    # cv2.imwrite('mp-gting.jpg', result)

    # print(human_detect(image))

    bucket = 'weishemg'
    s3 = boto3.client('s3')

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8080, type=int)
    args = vars(parser.parse_args())

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = VideoServicer()
    video_pb2_grpc.add_VideoProcessorServicer_to_server(servicer, server)


    try:
        q = multiprocess.Queue(maxsize=300)
        q2 = multiprocess.Queue(maxsize=10)

        p1 = multiprocess.Process(target=gstreamer_camera, args=(q, ))
        p2 = multiprocess.Process(target=gstreamer_rtmpstream, args=(q,))

        p1.start()
        p2.start()

        server.add_insecure_port(f"{args['ip']}:{args['port']}")
        server.start()
        print(f"Run gRPC Server at {args['ip']}:{args['port']}")
        server.wait_for_termination()
        p1.join()
        p2.join()

    except KeyboardInterrupt:
        pass



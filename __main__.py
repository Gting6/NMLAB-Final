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
import requests
import random
from paho.mqtt import client as mqtt_client


def gstreamer_camera(queue):
    # Use the provided pipeline to construct the video capture in opencv
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

def gstreamer_rtmpstream(queue):
    # Use the provided pipeline to construct the video writer in opencv
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
        'rtmpsink location="rtmp://a.rtmp.youtube.com/live2/jvmm-p5zf-79ux-ywtq-a9pp app=live2" audiotestsrc ! '
        "voaacenc bitrate=128000 ! "
        "mux. "
    )

    writer = cv2.VideoWriter(pipeline2, 0, 30.0, (1280, 720))
    print("?")
    algorithm = 0
    cnt = 0
    while True:
        frame = queue.get()
        cnt += 1
        if not q2.empty():
            algorithm = eval(q2.get())['command']  
            # print(type(algorithm))

            # print("setting alg to", algorithm['command'])
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
        elif algorithm == "snapshot":
            s = str(int(time.time()))
            print("uploading ..." + s)
            aws_upload(frame, s)
            print("uploaded! Now calling api")
            obj = {"id":"gting0906", "img_url":("https://weishemg.s3.ap-northeast-1.amazonaws.com/"+s+".PNG")}
            _ = requests.post(url, data =obj)
            algorithm = 0
        else: 
            pass
        # if cnt == 100:   
        #     print(human_detect(frame))
        #     cnt = 0

def human_detect(image):
    mp_object_detection = mp.solutions.object_detection
    # For static images:

    with mp_object_detection.ObjectDetection(
        min_detection_confidence=0.1) as object_detection:

        results = object_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for i in range(len(results.detections)):
                if "person" in results.detections[i].label:
                    # if results.detections[i].score > 0.5:
                    if results.detections[i].score[0] > 0.5:
                        print("Hi Human")
                        print(results.detections[i].score[0])
                    if results.detections[i].score[0] > 0.5:
                        s = str(int(time.time()))
                        print("uploading ..." + s)
                        tmp = 0
                        for i in range(5):
                            if not q.empty():
                                frame = q.get(0)
                            else:
                                break
                        aws_upload(frame, s)
                        print("uploaded! Now calling api")
                        obj = {"id":"gting0906", "img_url":("https://weishemg.s3.ap-northeast-1.amazonaws.com/"+s+".PNG")}
                        x = requests.post(url, data =obj)
                        print("Calling api result:", x)
                    return 1
                else:
                    pass
                    # print("No Human")
        return 0

def aws_upload(image, name):

    # convert numpy array to image first
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image).convert('RGB')
    buffer = BytesIO()
    img.save(buffer, format='png')
    buffer.seek(0)  
    key = name + ".PNG"
    sent_data = s3.put_object(Bucket=bucket, Key=key, Body=buffer,ContentType='image/png')
    if sent_data['ResponseMetadata']['HTTPStatusCode'] != 200:
        print("Error when upload!")
    return "https://weishemg.s3.ap-northeast-1.amazonaws.com/" + key

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

def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def subscribe(client: mqtt_client, queue):
    def on_message(client, userdata, msg):
        print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
        queue.put(msg.payload.decode())

    client.subscribe(topic)
    client.on_message = on_message


def run(queue):
    print("running process 3")
    client = connect_mqtt()
    subscribe(client, queue)
    client.loop_forever()



if __name__ == "__main__":
    bucket = 'weishemg'
    s3 = boto3.client('s3')
    broker = 'broker.emqx.io'
    port = 1883
    topic = "c2fd964cd38b477fa55c6d15ac8a8df70557cad8"
    url = 'https://nmlab-securitycam.herokuapp.com/api/alert'
    # generate client ID with pub prefix randomly
    client_id = f'python-mqtt-{random.randint(0, 100)}'
    username = 'emqx'
    password = 'public'

    try:
        q = multiprocess.Queue(maxsize=300)
        q2 = multiprocess.Queue(maxsize=10)

        p1 = multiprocess.Process(target=gstreamer_camera, args=(q, ))
        p2 = multiprocess.Process(target=gstreamer_rtmpstream, args=(q,))
        p3 = multiprocess.Process(target=run, args=(q2,))

        p1.start()
        p2.start()
        p3.start()

        p1.join()
        p2.join()
        p3.join()

    except KeyboardInterrupt:
        pass



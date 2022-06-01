import grpc
import argparse
import os
import os.path as osp
import sys
BUILD_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "build/service/")
sys.path.insert(0, BUILD_DIR)
import video_pb2_grpc
import video_pb2


def main(args):
    host = f"{args['ip']}:{args['port']}"
    print(host)
    # construct a connection to server
    with grpc.insecure_channel(host) as channel:
        stub = video_pb2_grpc.VideoProcessorStub(channel)

        request = video_pb2.VideoRequest()
        request.algorithm = args['order']

        response = stub.Compute(request)
        print(response.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--order", type=int, default=10)
    args = vars(parser.parse_args())
    main(args)

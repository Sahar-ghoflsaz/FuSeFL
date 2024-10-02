import socket
import argparse
import os
import signal
import subprocess
import time

import torch
import torch.optim as optim
import torch.nn as nn
import copy
torch.set_num_threads(1)

import syft
import syft as sy
from syft.serde.compression import NO_COMPRESSION
from syft.serde import serialize, deserialize
from syft.grid.clients.data_centric_fl_client import DataCentricFLClient

sy.serde.compression.default_compress_scheme = NO_COMPRESSION

from procedure import train, test
from data import get_data_loaders, get_number_classes, get_federated_data_loaders
from models import get_model, load_state_dict
from preprocess import build_prepocessing

from syft import FixedPrecisionTensor

import requests
import json

# Define the server's endpoint where the JSON data is available
url = 'http://127.0.0.1:5000/send_model'



NUM_CLIENTS = 16
BATCH_SIZE = 64
NUM_GROUPS = int(NUM_CLIENTS / 2)

SERVER = 'localhost'
PORT = 5000

def client_socket_start(host, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    print(f"started {host}:{port}")
    return client_socket

def receive_client1(client_socket):
    serial_new = b""
    while True:
        print("more data recieved")
        data = client_socket.recv(4096)
        # if not data:
        if b"<END>" in data:
            serial_new += data.replace(b"<END>", b"")
            break
        serial_new += data

    print("encrypted model received")
    
    # received_state_dict = deserialize(serial_new)
    # # out_features = get_number_classes(dataset)
    # recieved_model = get_model(model,dataset, out_features=get_number_classes(dataset))
    # received_state_dict = convert_fixed_precision(received_state_dict)
    # recieved_model.load_state_dict(received_state_dict)

    # print(recieved_model)
    # newmodel.load_state_dict(recieved_model)
    # print(newmodel)
    return serial_new
# app = Flask(__name__)
def receive_client2(client_socket):
    serial_new = b""
    while True:
        print("more data recieved")
        data = client_socket.recv(32)
        # if not data:
        if b"<END>" in data:
            serial_new += data.replace(b"<END>", b"")
            break
        serial_new += data

    print("encrypted key received")

    return serial_new

def receive_client3(client_socket):
    serial_new = b""
    while True:
        print("more data recieved")
        data = client_socket.recv(1)
        # if not data:
        if b"<END>" in data:
            serial_new += data.replace(b"<END>", b"")
            break
        serial_new += data

    print("encrypted public received")
    return serial_new

def receive_workers(client_socket):
    serial_new = b""
    while True:
        print("more data recieved")
        data = client_socket.recv(2048)
        # if not data:
        if b"<END>" in data:
            serial_new += data.replace(b"<END>", b"")
            break
        serial_new += data

    print("encrypted public received")
    return serial_new

def send_model_toserver(client_socket, model):
    # serial_model = serialize(model)
    client_socket.sendall(model)
    client_socket.sendall(b"<END>")
    print("Model sent to server")



def start_client(args, global_model, local_model, globalClients, gepoch,kwargs):
    # print("ok")
    # client_socket = client_socket_start(SERVER, PORT)
    # receive_client(client_socket, model)
    # send_model_toserver(client_socket, model)
    # federated_train_loaders, new_test_loaders = get_federated_data_loaders(args, kwargs[globalClients], num_clients=NUM_CLIENTS, private=True)
    # if args.test and not args.train:
    #     load_state_dict(local_model[gepoch*NUM_GROUPS + globalClients], args.model, args.dataset)
    # local_model[gepoch*NUM_GROUPS + globalClients].eval()
    # if torch.cuda.is_available():
    #     sy.cuda_force = True


    # for localClients in range(2):
        
    #     client = globalClients * 2 + localClients
    #     print("running training for the data of client" + str(client))
    #     # print("\n before:\n")

    #     # print(model[gepoch*NUM_GROUPS + globalClients].fc1.weight.data)

    #     # print(local_model[gepoch*NUM_GROUPS + globalClients].fc1.weight.data)
    #     local_model[gepoch*NUM_GROUPS + globalClients].decrypt()
    #     print("start local model partial")
    #     print(local_model[gepoch*NUM_GROUPS+globalClients].fc1.weight.data)
    #     if not args.public:
    #         local_model[gepoch*NUM_GROUPS + globalClients].encrypt(**kwargs[globalClients])
    #         if args.fp_only:  # Just keep the (Autograd+) Fixed Precision feature
    #             local_model[gepoch*NUM_GROUPS + globalClients].get()

    #     if args.train:
    #         for epoch in range(args.local_epochs):
    #             optimizer = optim.SGD(local_model[gepoch*NUM_GROUPS + globalClients].parameters(), lr=args.lr, momentum=args.momentum)

    #             if not args.public:
    #                 optimizer = optimizer.fix_precision(
    #                     precision_fractional=args.precision_fractional, dtype=args.dtype
    #                 )
    #             print("start train")
    #             train_time = train(args,local_model[gepoch*NUM_GROUPS + globalClients], federated_train_loaders[client], optimizer, epoch)
    #             print("end train")
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        help="model to use for inference (network1, network2, lenet, alexnet, vgg16, resnet18)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset to use (mnist, cifar10, hymenoptera, tiny-imagenet)",
    )
    parser.add_argument(
        "--public",
        help="[needs --train] Train without fix precision or secret sharing",
        action="store_true",
    )

    cmd_args = parser.parse_args()
    model = cmd_args.model.lower()
    dataset = cmd_args.dataset.lower()
    public = cmd_args.public

    hook = sy.TorchHook(torch)
    jack = sy.VirtualWorker(hook, id="jack")
    john = sy.VirtualWorker(hook, id="john")
    crypto_provider1 = sy.VirtualWorker(hook, id="crypto_provider")
    workers_a = [jack, john]
    sy.local_worker.clients = workers_a
    encryption_kwargs1 = dict(
        workers=workers_a, crypto_provider=crypto_provider1, protocol="fss"
    )
    kwargs1 = dict(
        requires_grad=True,
        precision_fractional=5,
        dtype="long",
        **encryption_kwargs1,
    )
    
    totModel = None
    modeltot = get_model(model, dataset, out_features=get_number_classes(dataset))


    
        # if args.fp_only:  # Just keep the (Autograd+) Fixed Precision feature
        #     global_model.get()
    client_socket = client_socket_start(SERVER, PORT)
    print("client socket done")

    # workers_ser = receive_workers(client_socket)
    # workers = deserialize(workers_ser)
   
    print("client workers done")

    newmodel = None
    # kwargs1 = None
    newmodel = receive_client1(client_socket)
    # kwargs1ser = receive_client2(client_socket)
    # publicser = receive_client3(client_socket)
    # public = deserialize(publicser)
    # kwargs1 = deserialize(kwargs1ser)
    # #new
    # print("kw ", kwargs1)
    print("start receiving model")
    received_state_dict = deserialize(newmodel)
    hi = received_state_dict
    print("done receiving")
    # if not public:
    print(modeltot)
    # print("sahar")
    modeltot.encrypt(**kwargs1)
    # print("hi = ", hi)
    print(modeltot)

    modeltot.load_state_dict(hi)
    # #new
    print("client data recieved")
    send_model_toserver(client_socket, newmodel)   
    print("client data sent to server")   
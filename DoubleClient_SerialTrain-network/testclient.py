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

NUM_CLIENTS = 16
BATCH_SIZE = 64
NUM_GROUPS = int(NUM_CLIENTS / 2)

SERVER = 'localhost'
PORT = 8080

def client_socket_start(host, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    print(f"started {host}:{port}")
    return client_socket

def receive_client(client_socket):
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

    recieved_model = deserialize(serial_new)
    # newmodel.load_state_dict(recieved_model)
    return recieved_model

def send_model_toserver(client_socket, model):
    serial_model = serialize(model.state_dict())
    client_socket.sendall(serial_model)
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
    hook = sy.TorchHook(torch)
    jack = sy.VirtualWorker(hook, id="jack")
    john = sy.VirtualWorker(hook, id="john")
    crypto_provider1 = sy.VirtualWorker(hook, id="crypto_provider")
    workers_a = [jack, john]
    sy.local_worker.clients = workers_a
    # encryption_kwargs1 = dict(
    #     workers=workers_a, crypto_provider=crypto_provider1, protocol=args.protocol
    # )
    # kwargs1 = dict(
    #     requires_grad=args.requires_grad,
    #     precision_fractional=args.precision_fractional,
    #     dtype=args.dtype,
    #     **encryption_kwargs1,
    # )

    client_socket = client_socket_start(SERVER, PORT)
    print("client socket done")
    model = ""
    model = receive_client(client_socket) 
    print("client data recieved")
    send_model_toserver(client_socket, model)   
    print("client data sent to server")   
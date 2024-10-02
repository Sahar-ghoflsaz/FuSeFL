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
from client import start_client
# from main import main_model

from flask import Flask, request
import pickle
import json
import time

NUM_CLIENTS = 16
BATCH_SIZE = 64
NUM_GROUPS = int(NUM_CLIENTS / 2)

#LAN connection
HOST = '0.0.0.0'
PORT = 5000

# app = Flask(__name__)

def server_socket_start(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"listen {host}:{port}")
    return server_socket

# def get_model_mine():
#     model = main_model()
#     return model
    
# def main_model():
#     hook = sy.TorchHook(torch)
#     totModel = None
#     modeltot = get_model(args.model, args.dataset, out_features=get_number_classes(args.dataset))
#     jack = sy.VirtualWorker(hook, id="jack")
#     john = sy.VirtualWorker(hook, id="john")
#     crypto_provider1 = sy.VirtualWorker(hook, id="crypto_provider")
#     workers_a = [jack, john]
#     sy.local_worker.clients = workers_a
#     encryption_kwargs1 = dict(
#         workers=workers_a, crypto_provider=crypto_provider1, protocol=args.protocol
#     )
#     kwargs1 = dict(
#         requires_grad=args.requires_grad,
#         precision_fractional=args.precision_fractional,
#         dtype=args.dtype,
#         **encryption_kwargs1,
#     )
#     global_model = copy.deepcopy(modeltot)
#     if not args.public:
#         global_model.encrypt(**kwargs1)
#         if args.fp_only:  # Just keep the (Autograd+) Fixed Precision feature
#             global_model.get()
#     print("sahar")
#     global_model.load_state_dict(global_model.state_dict())
#     print("sahar2")
#     print(global_model.state_dict())
    
#     return modeltot

# @app.route('/send_model', methods=['GET'])
def send_model_client(connection, model):
    serial_model = serialize(model.state_dict())
    connection.sendall(serial_model)
    connection.sendall(b"<END>")

def send_key_client(connection, kwargs1):
    print("kw ", kwargs1)
    serialkey = serialize(kwargs1)
    connection.sendall(serialkey)
    connection.sendall(b"<END>")

def send_workers_client(connection, workers):
    serial = serialize(workers)
    connection.sendall(serial)
    connection.sendall(b"<END>")

def start_server(args,workers, global_model, kwargs1, local_model,kwargs, gepoch):
    # public_train_loader, public_test_loader = get_data_loaders(args, kwargs, private=False)
    # model = [None] * NUM_CLIENTS * args.global_epochs
    # for gepoch in range(args.global_epqochs):

    global_model.decrypt()
    recieved_model = copy.deepcopy(global_model)
    if not args.public:
        global_model.encrypt(**kwargs1)
        recieved_model.encrypt(**kwargs1)
        if args.fp_only:  # Just keep the (Autograd+) Fixed Precision feature
            global_model.get()
            recieved_model.get()

    print("running training global epoch") # + str(gepoch))

    # I need to request the connection here
    server_socket = server_socket_start(HOST, PORT)
    print("socket done")
    connection, address = server_socket.accept()
    print("connection accepted")
    # send_workers_client(connection, workers)
    # print("workers sent to client")
    # time.sleep(1)
    # app.run(debug=True, host='0.0.0.0', port=5000)
    print("kw= ", kwargs1)
    send_model_client(connection, global_model)
    print("model sent to client")
    time.sleep(1)

    # send_key_client(connection, kwargs1)
    # print("key sent to client")
    # time.sleep(1)
    # send_pub_client(connection, args.public)
    # print("public sent to client")


    serial_new = b""
    while True:
        data = connection.recv(4096)
        if b"<END>" in data:
            serial_new += data.replace(b"<END>", b"")
            break
        serial_new += data

    print("encrypted model received")
    received_state_dict = deserialize(serial_new)
    # out_features = get_number_classes(args.dataset)

    # # Create a new instance of the model with the correct dataset and out_features
    # recieved_model = get_model(args.model, args.dataset, out_features)
    # recieved_model = deserialize(serial_new)
    hi = received_state_dict
    recieved_model.load_state_dict(hi)
    # newmodel.load_state_dict(recieved_model)
    print("model received from client")
    # Since for now i am simulating I just use client file
    # for globalClients in range(NUM_GROUPS):

    # print("running training on group" + str(globalClients))
    # start_client(args,global_model, local_model, globalClients, gepoch,kwargs)
    
    # local_model[gepoch*NUM_GROUPS+globalClients].decrypt()
    # print("local model final")
    # print(local_model[gepoch*NUM_GROUPS+globalClients].fc1.weight.data)
    # if not args.public:
    #     local_model[gepoch*NUM_GROUPS+globalClients].encrypt(**kwargs1)
    #     if args.fp_only:  
    #         local_model[gepoch*NUM_GROUPS+globalClients].get()

    # if globalClients == 0:
        
    #     global_model.fc1.weight.data = local_model[gepoch*NUM_GROUPS+globalClients].fc1.weight.data
    #     global_model.fc2.weight.data = local_model[gepoch*NUM_GROUPS+globalClients].fc2.weight.data
    #     global_model.fc3.weight.data = local_model[gepoch*NUM_GROUPS+globalClients].fc3.weight.data

    # else: 
    #     global_model.fc1.weight.data += local_model[gepoch*NUM_GROUPS+globalClients].fc1.weight.data
    #     global_model.fc2.weight.data += local_model[gepoch*NUM_GROUPS+globalClients].fc2.weight.data
    #     global_model.fc3.weight.data += local_model[gepoch*NUM_GROUPS+globalClients].fc3.weight.data

    # if globalClients ==(NUM_GROUPS-1):
    #     global_model.fc1.weight.data /= NUM_GROUPS
    #     global_model.fc2.weight.data /= NUM_GROUPS
    #     global_model.fc3.weight.data /= NUM_GROUPS

            
    # print("evaluation")
    # global_model.decrypt()
    # print("global model final")
    # print(global_model.fc1.weight.data)
    # if not args.public:
    #     global_model.encrypt(**kwargs1)
    #     if args.fp_only:  # Just keep the (Autograd+) Fixed Precision feature
    #         global_model.get()
    # private_train_loader, private_test_loader = get_data_loaders(args, kwargs1, private=True)
    # test_time, accuracy = test(args, global_model, private_test_loader)

    # print("evaluation")
    # print(recieved_model.fc1.weight.data)
    recieved_model.decrypt()
    global_model.decrypt()
    print("new model final")
    print(recieved_model.fc1.weight.data)
    print("global")
    print(global_model.fc1.weight.data)
    # if not args.public:
    #     recieved_model.encrypt(**kwargs1)
    #     if args.fp_only:  # Just keep the (Autograd+) Fixed Precision feature
    #         recieved_model.get()
    connection.close()
    server_socket.close()

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
        "--batch_size",
        type=int,
        help="size of the batch to use. Default 128.",
        default=64,
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        help="size of the batch to use",
        default=None,
    )

    parser.add_argument(
        "--preprocess",
        help="[only for speed test] preprocess data or not",
        action="store_true",
    )

    parser.add_argument(
        "--fp_only",
        help="Don't secret share values, just convert them to fix precision",
        action="store_true",
    )

    parser.add_argument(
        "--public",
        help="[needs --train] Train without fix precision or secret sharing",
        action="store_true",
    )

    parser.add_argument(
        "--test",
        help="run testing on the complete test dataset",
        action="store_true",
    )

    parser.add_argument(
        "--train",
        help="run training for n epochs",
        action="store_true",
    )

    parser.add_argument(
        "--gepochs",
        type=int,
        help="[needs --train] number of global epochs to train on. Default 15.",
        default=2,
    )
    parser.add_argument(
        "--lepochs",
        type=int,
        help="[needs --train] number of local epochs to train on. Default 15.",
        default=8,
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="[needs --train] learning rate of the SGD. Default 0.01.",
        default=0.01,
    )

    parser.add_argument(
        "--momentum",
        type=float,
        help="[needs --train] momentum of the SGD. Default 0.9.",
        default=0.9,
    )

    parser.add_argument(
        "--websockets",
        help="use PyGrid nodes instead of a virtual network. (nodes are launched automatically)",
        action="store_true",
    )

    parser.add_argument(
        "--verbose",
        help="show extra information and metrics",
        action="store_true",
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        help="[needs --test or --train] log intermediate metrics every n batches. Default 10.",
        default=10,
    )

    parser.add_argument(
        "--comm_info",
        help="Print communication information",
        action="store_true",
    )

    parser.add_argument(
        "--pyarrow_info",
        help="print information about PyArrow usage and failure",
        action="store_true",
    )

    cmd_args = parser.parse_args()

    # Sanity checks

    if cmd_args.test or cmd_args.train:
        assert (
            not cmd_args.preprocess
        ), "Can't preprocess for a full epoch evaluation or training, remove --preprocess"

    if cmd_args.train:
        assert not cmd_args.test, "Can't set --test if you already have --train"

    if cmd_args.fp_only:
        assert not cmd_args.preprocess, "Can't have --preprocess in a fixed precision setting"
        assert not cmd_args.public, "Can't have simultaneously --fp_only and --public"

    if not cmd_args.train:
        assert not cmd_args.public, "--public is used only for training"

    if cmd_args.pyarrow_info:
        sy.pyarrow_info = True

    class Arguments:
        model = cmd_args.model.lower()
        dataset = cmd_args.dataset.lower()
        preprocess = cmd_args.preprocess
        websockets = cmd_args.websockets
        verbose = cmd_args.verbose

        train = cmd_args.train
        n_train_items = -1 if cmd_args.train else cmd_args.batch_size
        test = cmd_args.test or cmd_args.train
        n_test_items = -1 if cmd_args.test or cmd_args.train else cmd_args.batch_size

        batch_size = cmd_args.batch_size
        # Defaults to the train batch_size
        test_batch_size = cmd_args.test_batch_size or cmd_args.batch_size

        log_interval = cmd_args.log_interval
        comm_info = cmd_args.comm_info

        local_epochs = cmd_args.lepochs
        global_epochs = cmd_args.gepochs
        lr = cmd_args.lr
        momentum = cmd_args.momentum

        public = cmd_args.public
        fp_only = cmd_args.fp_only
        requires_grad = cmd_args.train
        dtype = "long"
        protocol = "fss"
        precision_fractional = 5 if cmd_args.train else 4

    args = Arguments()

    if args.websockets:
        print("Launching the websocket workers...")

        def kill_processes(worker_processes):
            for worker_process in worker_processes:
                pid = worker_process.pid
                try:
                    os.killpg(os.getpgid(worker_process.pid), signal.SIGTERM)
                    print(f"Process {pid} killed")
                except ProcessLookupError:
                    print(f"COULD NOT KILL PROCESS {pid}")

        worker_processes = [
            subprocess.Popen(
                f"./scripts/launch_{worker}.sh",
                stdout=subprocess.PIPE,
                shell=True,
                preexec_fn=os.setsid,
                executable="/bin/bash",
            )
            for worker in ["alice", "bob", "crypto_provider"]
        ]
        time.sleep(7)
        try:
            print("LAUNCHED", *[p.pid for p in worker_processes])
            run(args)
            kill_processes(worker_processes)
        except Exception as e:
            kill_processes(worker_processes)
            raise e

    else:
        # arg= [None] * NUM_CLIENTS
        # totalModel = Null
        # for client in range(NUM_CLIENTS):
            # arg[client] = Arguments()
        # for client in range(NUM_CLIENTS):
            # print("running training on client" + str(client))
        run(args)
            # print("done on client" + str(client))
            # totalModel += arg[client].model
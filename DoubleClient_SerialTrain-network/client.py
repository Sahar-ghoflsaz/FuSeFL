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
from syft.grid.clients.data_centric_fl_client import DataCentricFLClient

sy.serde.compression.default_compress_scheme = NO_COMPRESSION

from procedure import train, test
from data import get_data_loaders, get_number_classes, get_federated_data_loaders
from models import get_model, load_state_dict
from preprocess import build_prepocessing

NUM_CLIENTS = 16
BATCH_SIZE = 64
NUM_GROUPS = int(NUM_CLIENTS / 2)

def start_client(args, global_model, local_model, globalClients, gepoch,kwargs):
    # print("ok")
    federated_train_loaders, new_test_loaders = get_federated_data_loaders(args, kwargs[globalClients], num_clients=NUM_CLIENTS, private=True)
    if args.test and not args.train:
        load_state_dict(local_model[gepoch*NUM_GROUPS + globalClients], args.model, args.dataset)
    local_model[gepoch*NUM_GROUPS + globalClients].eval()
    if torch.cuda.is_available():
        sy.cuda_force = True


    for localClients in range(2):
        
        client = globalClients * 2 + localClients
        print("running training for the data of client" + str(client))
        # print("\n before:\n")

        # print(model[gepoch*NUM_GROUPS + globalClients].fc1.weight.data)

        # print(local_model[gepoch*NUM_GROUPS + globalClients].fc1.weight.data)
        local_model[gepoch*NUM_GROUPS + globalClients].decrypt()
        print("start local model partial")
        print(local_model[gepoch*NUM_GROUPS+globalClients].fc1.weight.data)
        if not args.public:
            local_model[gepoch*NUM_GROUPS + globalClients].encrypt(**kwargs[globalClients])
            if args.fp_only:  # Just keep the (Autograd+) Fixed Precision feature
                local_model[gepoch*NUM_GROUPS + globalClients].get()

        if args.train:
            for epoch in range(args.local_epochs):
                optimizer = optim.SGD(local_model[gepoch*NUM_GROUPS + globalClients].parameters(), lr=args.lr, momentum=args.momentum)

                if not args.public:
                    optimizer = optimizer.fix_precision(
                        precision_fractional=args.precision_fractional, dtype=args.dtype
                    )
                print("start train")
                train_time = train(args,local_model[gepoch*NUM_GROUPS + globalClients], federated_train_loaders[client], optimizer, epoch)
                print("end train")
    
        # if localClients == 0:

        #     # local_model[gepoch*NUM_GROUPS + globalClients].decrypt()
        #     print("local_model 0")
        #     # print(model[gepoch*NUM_GROUPS + globalClients].fc1.weight.data)
        #     # print("alice has: " + str(alice._objects))
        #     # model[client].fc1.weight.data=model[client].fc1.weight.data.send(john)

        # else:

        #     local_model[gepoch*NUM_GROUPS + globalClients].decrypt()
        #     print("local_model 1")
        #     print(local_model[gepoch*NUM_GROUPS + globalClients].fc1.weight.data)
        #     local_model = copy.deepcopy(local_model[gepoch*NUM_GROUPS + globalClients])
        #     if not args.public:
        #         # model[client].encrypt(**kwargs[client])
        #         local_model.encrypt(**kwargs1)
        #         if args.fp_only:  # Just keep the (Autograd+) Fixed Precision feature
        #             # model[client].get()
        #             local_model.get()

            # print("alice has: " + str(alice._objects))
            # local_model.fc1.weight.data = (local_model.fc1.weight.data + new_model.fc1.weight.data)
            # local_model.fc2.weight.data = (local_model.fc2.weight.data + new_model.fc2.weight.data)
            # local_model.fc3.weight.data = (local_model.fc3.weight.data + new_model.fc3.weight.data)


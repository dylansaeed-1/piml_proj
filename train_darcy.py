import torch
import torch.nn as nn
import numpy as np
from fno_2D import *
from lploss import *
import grpc
import darcy_pb2
import darcy_pb2_grpc

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda:0")
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    
def main():

    mode1 = 10
    mode2 = 10
    width = 36

    model = Net2d(mode1, mode2, width)
    model.to(device)

    epochs = 1 # Change once it works
    learning_rate = 0.001
    scheduler_step = 2
    scheduler_gamma = 0.9
    learning_rate = 10e-3
    time_steps = 1
    dim_x, dim_y = (64, 64)
    init_pressure = torch.tensor(np.zeros((1,dim_x, dim_y, 1)), dtype=torch.float32) #Should be (batch_size, x, y, t, c)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    myloss = LpLoss(size_average=False)

    curr_state = init_pressure
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = darcy_pb2_grpc.pimlStub(channel)
    for ep in range(epochs):
        ##Calculate next time step and send to grpc client to get residual
        curr_state = init_pressure.to(device)
        print(curr_state.dtype)
        for t in range(time_steps):
            next_state = model(curr_state)
        

    return 0

if __name__ == "__main__":
    main()
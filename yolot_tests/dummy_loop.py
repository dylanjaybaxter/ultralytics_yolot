import torch
from torch import nn
import torch.optim as opt
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

# Warm-up and training loop
total_epochs = 5
epoch_iters = 10
warmup_steps = 5

lr0 = 0.01
lrf = 0.0003

model = nn.Sequential(
    nn.Linear(5, 5),
    nn.ReLU(),
    nn.Linear(5, 5)
)

# Example optimizer
optimizer = opt.SGD(model.parameters(), lr=0.01, momentum=0.9)
lam1 = lambda epoch: max((0.9 ** epoch), lrf/lr0)

# Example scheduler
scheduler = LambdaLR(optimizer, lr_lambda=[lam1])

lr0=100

warmup_counter = 0
for epoch in range(total_epochs):
    for iter in range(epoch_iters):
        # Warm-up phase
        if warmup_steps > warmup_counter:
            for idx, x in enumerate(optimizer.param_groups):
                x['lr'] = np.interp(warmup_counter, [0,warmup_steps], [0.0, lr0])
                if "momentum" in x: 
                    x["momentum"] = np.interp(warmup_counter, [0,warmup_steps], [0.8, 0.9])
            warmup = True
            warmup_counter += 1
        else:
            warmup = False
        print("Epoch {}, W:{}, Learning Rate: {}".format(epoch, warmup_counter, optimizer.param_groups[0]['lr']))
    scheduler.step()
        
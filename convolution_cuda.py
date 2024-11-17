import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
from torchvision import datasets,transforms

import torch.multiprocessing as mp
import torch.distributed as dist

import os
import time
import subprocess



class Convolution(nn.Module):
    def __init__(self):
        super(Convolution,self).__init__()
        self.cn1=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1)
        self.cn2=nn.Conv2d(in_channels=16,out_channels=32,stride=1,kernel_size=3)
        self.dropout1=nn.Dropout(0.25)
        self.dropout2=nn.Dropout(0.10)
        self.fc1=nn.Linear(4608,64)
        self.fc2=nn.Linear(64,10)

    def forward(self,x):
        x=self.cn1(x)
        x=F.relu(x)
        x=self.cn2(x)
        x=F.relu(x)
        x= x=F.max_pool2d(x,2)
        x=self.dropout1(x)
        x=torch.flatten(x,1)
        x=self.fc1(x)
        x=F.relu(x)
        x=self.dropout2(x)
        x=self.fc2(x)
        
        x = F.log_softmax(x, dim=1)
        return x
def train(gpu_name, args):
    rank = args.machine_id * args.num_gpu_processes + gpu_name
    dist.init_process_group(
        backend='gloo',
        init_method="env://?use_libuv=False",
        world_size=args.world_size,
        rank=rank
    )
    torch.manual_seed(0)
    
    # Move model to the appropriate GPU
    model = Convolution().to(gpu_name)
    torch.cuda.set_device(gpu_name)
    criterion = nn.NLLLoss().cuda(gpu_name)

    # Prepare dataset
    train_dataset = datasets.MNIST('data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1302,), (0.3069,))
                                   ]))  
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
    train_dataloader = torch.utils.data.DataLoader(
       dataset=train_dataset,
       batch_size=args.batch_size,
       shuffle=False,            
       num_workers=4,
       pin_memory=True,
       sampler=train_sampler
    )
    optimizer = optim.Adadelta(model.parameters(), lr=0.5)
    
    # Wrap model in DistributedDataParallel
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_name])

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        for b_i, (X, y) in enumerate(train_dataloader):
            X, y = X.cuda(gpu_name, non_blocking=True), y.cuda(gpu_name, non_blocking=True)
            with torch.cuda.amp.autocast():
                pred_prob = model(X)
                loss = criterion(pred_prob, y) 
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if b_i % 10 == 0 and gpu_name == 0:
                print('epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}'.format(
                    epoch, b_i, len(train_dataloader),
                    100. * b_i / len(train_dataloader), loss.item()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--num-machines",default=1,type=int,)
    parser.add_argument('--num-gpu-processes', default=1, type=int)
    parser.add_argument('--machine-id', default=0, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    args = parser.parse_args()
    
    args.world_size = args.num_gpu_processes * args.num_machines                
    os.environ['MASTER_ADDR'] = '127.0.0.1'              
    os.environ['MASTER_PORT'] = '8892'      
    start = time.time()
    mp.spawn(train, nprocs=args.num_gpu_processes, args=(args,))
    print(f"Finished training in {time.time()-start} secs")
if __name__ == '__main__':
    main()
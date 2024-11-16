import torch 
import torch.utils.data.distributed
from torchvision import datasets,transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import time
import argparse
 
class Convolution(nn.Module):
    def __init__(self):
        super(Convolution,self).__init__()
        self.cn1=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1)
        self.cn2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1)
        self.dropout1=nn.Dropout(0.10)
        self.dropout2=nn.Dropout(0.25)
        self.fc1=nn.Linear(4608,64)
        self.fc2=nn.Linear(64,10)
    def forward(self,x):
        x=self.cn1(x)##(28+2(0)-3/1)+1=(1,16,26,26)
        x=F.relu(x)##(28+2(0)-3/1)+1=(1,16,26,26)
        x=self.cn2(x)##(26+2(0)-3/1)+1=(1,32,24,24)
        x=F.relu(x)##(26+2(0)-3/1)+1=(1,32,24,24)
        x=F.max_pool2d(x,2)##(24-2/2)+1=(1,32,12,12)
        x=self.dropout1(x)##(24-2/2)+1=(1,32,12,12)
        x=torch.flatten(x,1)##(32*12*12)=(1*4608)
        x=self.fc1(x)##(1,64)
        x=F.relu(x)##(1,64)
        x=self.dropout2(x)##(64,10)
        x=self.fc2(x)##(64,10)
        
        x = F.log_softmax(x, dim=1)##(1,10)
        return x
def train(cpu_num, args):
    rank = args.machine_id * args.num_processes + cpu_num                        
    dist.init_process_group(                                   
    backend='gloo',                                         
    init_method='env://?use_libuv=False',                                   
    world_size=args.world_size,                              
    rank=rank                                               
    ) 
    torch.manual_seed(0)
    device = torch.device("cpu")
    train_dataset = datasets.MNIST('data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1302,), (0.3069,))]))  
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
    train_dataloader = torch.utils.data.DataLoader(
       dataset=train_dataset,
       batch_size=args.batch_size,
       shuffle=False,            
       num_workers=0,
       sampler=train_sampler)
    model = Convolution()
    optimizer = optim.Adadelta(model.parameters(), lr=0.5)
    model = nn.parallel.DistributedDataParallel(model)
    model.train()
    for epoch in range(args.epochs):
        for b_i, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            pred_prob = model(X)
            loss = F.nll_loss(pred_prob, y) # nll is the negative likelihood loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if b_i % 10 == 0 and cpu_num==0:
                print('epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}'.format(
                    epoch, b_i, len(train_dataloader),
                    100. * b_i / len(train_dataloader), loss.item()))
def main():
        parser=argparse.ArgumentParser()
        parser.add_argument('--num-machines',default=1,type=int,)
        parser.add_argument('--num-processes', default=1, type=int)
        parser.add_argument('--machine-id', default=0, type=int)
        parser.add_argument('--epochs', default=1, type=int)
        parser.add_argument('--batch-size', default=128, type=int)
        args=parser.parse_args()
        args.world_size=args.num_processes*args.num_machines
        os.environ['MASTER_ADDR'] = '127.0.0.1'              
        os.environ['MASTER_PORT'] = '8892'    
        start = time.time()
        mp.spawn(train, nprocs=args.num_processes, args=(args,))
        print(f"Finished training in {time.time()-start} secs")
if __name__ == '__main__':
    main()
    


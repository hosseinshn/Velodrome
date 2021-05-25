import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def Network(args, X):
    IE_dim = X.shape[1]

    class Net1(nn.Module):
        def __init__(self, args):
            super(Net1, self).__init__()

            self.features = torch.nn.Sequential(
                nn.Linear(IE_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(args.ldr),
                nn.Linear(512, 128),
                nn.Sigmoid()) 

        def forward(self, x):
            out = self.features(x)
            return out     
    
    class Net2(nn.Module):
        def __init__(self, args):
            super(Net2, self).__init__()

            self.features = torch.nn.Sequential(
                nn.Linear(IE_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(args.ldr),
                nn.Linear(256, 256),
                nn.Sigmoid()
                ) 

        def forward(self, x):
            out = self.features(x)
            return out            

    class Net3(nn.Module):
        def __init__(self, args):
            super(Net3, self).__init__()    

            self.features = torch.nn.Sequential(
                nn.Linear(IE_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(args.ldr),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(args.ldr),
                nn.Linear(128, 128),
                nn.Sigmoid()
                )                        

        def forward(self, x):
            out = self.features(x)
            return out

    class Net4(nn.Module):
        def __init__(self, args):
            super(Net4, self).__init__()    

            self.features = torch.nn.Sequential(
                nn.Linear(IE_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(args.ldr),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(args.ldr),
                nn.Linear(64, 64),
                nn.Sigmoid()
                )                        

        def forward(self, x):
            out = self.features(x)
            return out        
        
    if args.hd == 1:
        Model = Net1(args)
    elif args.hd == 2:
        Model = Net2(args)
    elif args.hd == 3:
        Model = Net3(args)
    elif args.hd == 4:
        Model = Net4(args)        

    class Pred(nn.Module):
        def __init__(self, args):
            super(Pred, self).__init__()
            if args.hd == 1: 
                dim = 128            
            if args.hd == 2: 
                dim = 256
            if args.hd == 3:
                dim = 128
            if args.hd == 4:
                dim = 64                
            self.pred = torch.nn.Sequential(
                nn.Linear(dim, 1)) 

        def forward(self, x):
            out = self.pred(x)
            return out     
    torch.manual_seed(args.seed)    
    Predict_1 = Pred(args)
    torch.manual_seed(args.seed*2)
    Predict_2 = Pred(args)
    torch.manual_seed(args.seed)
        
    return Model, Predict_1, Predict_2

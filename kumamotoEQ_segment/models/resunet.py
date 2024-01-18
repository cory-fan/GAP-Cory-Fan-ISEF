import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import models.pointnet2_utils as pointnet2_utils
import models.pointMLP as pointMLP
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation,farthest_point_sample
import torch

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def bknn(x, k):
    inner = 2*torch.bmm(x.transpose(2,1), x)
    xx = torch.sum(torch.square(x), dim=1, keepdim = True)
    pairwise_dist = xx - inner + xx.transpose(2, 1)
    inv_dist = (-1)*pairwise_dist
    
    idx = inv_dist.topk(k=k,dim=-1)[1]
    
    return idx

def bdist(a, b):
    
    #B,C,N and B,C,M to B,N,M
    
    B,_,_ = a.shape
    
    aa = torch.sum(a**2,dim=1,keepdim=True)
    bb = torch.sum(b**2,dim=1,keepdim=True)
    
    inner = -2*torch.matmul(a.transpose(1,2),b)
    
    pairwise_distance = aa.transpose(1,2)+bb+inner
    
    return pairwise_distance

def bdknn(a,b,k):
    
    pairwise_dist = bdist(a,b)
    
    inv_dist = (-1)*pairwise_dist
    
    idx = inv_dist.topk(k=k,dim=-1)[1]
    
    return idx
    

class AffineGeometry(nn.Module):
    
    def getLaplacian(self,xyz, k, feat):
        
        B,F,N = feat.shape
        
        kNeighbors = bknn(xyz[:,0:2,:],k)
        
        allLaplacians = []
        
        for batchn in range(kNeighbors.shape[0]):
            iKNeighbors = kNeighbors[batchn]
            localfeat = feat[batchn,:,torch.flatten(iKNeighbors)]
            localfeat = localfeat.view(-1,iKNeighbors.shape[0]*F)
            #localfeat = localfeat.view(N,-1,F)
            #print(feat[batchn].shape,localfeat.mean(dim=1).shape)
            #print(localfeat.shape)
            #print(localfeat.shape)
            allLaplacians.append(localfeat)
        return torch.stack(allLaplacians)
        
    def __init__(self,n_points, k_value, n_feat):
        super().__init__()
                
        self.k = k_value
        self.npoints = n_points
                                 
        self.nfeat = n_feat #C'
        
        self.affine = nn.Conv1d(self.k,1,1)
        self.transform = nn.Linear(self.nfeat,self.nfeat) #(B,N,C') to (B, C', N)
        self.batchNorm = nn.BatchNorm1d(n_feat)
        self.rl = nn.ReLU()
        
    def forward(self,xyz, feat): #(B,N,C) to #(B,N,C'), where C is point dimensionality and C' is geometric features
        B, C, N = xyz.shape
        _, F, _ = feat.shape
        
        affineFeat = self.getLaplacian(xyz,self.k,feat)
        #print(affineFeat.shape)
        
        affineFeat = self.affine(affineFeat)
        #print(xyz.shape,feat.shape,affineFeat.shape)
        affineFeat = affineFeat.view((B,N,F))
        affineFeat = self.transform(affineFeat)
        affineFeat = self.rl(self.batchNorm(affineFeat.transpose(1,2)))
        
        return affineFeat
    
class downsampleBlock(nn.Module):
    
    def getNeighborFeat(self,refxyz,newxyz,k,feat):
    
        B,F,N = feat.shape
        
        kNeighbors = bdknn(newxyz,refxyz,k)
        
        allNeighFeat = []
        
        for batchn in range(kNeighbors.shape[0]):
            iKNeighbors = kNeighbors[batchn]
            localfeat = feat[batchn,:,torch.flatten(iKNeighbors)]
            localfeat = localfeat.view(-1,iKNeighbors.shape[0]*F)
            #localfeat = localfeat.view(N,-1,F)
            #print(feat[batchn].shape,localfeat.mean(dim=1).shape)
            #print(localfeat.shape)
            #print(localfeat.shape)
            allNeighFeat.append(localfeat)
        return torch.stack(allNeighFeat)
    
    def __init__(self,in_point,out_point,in_channel,out_channel,radius):
        
        #input: B,C,N
        #output: B,C',N
        
        super().__init__()
        
        self.radius = radius
        self.k = 4
        
        self.in_point = in_point
        self.out_point = out_point
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.conv = pointMLP.ConvBNReLU1D(in_channel,out_channel)
        
        
    def forward(self,xyz,feat):
        
        B,F,N = feat.shape
        
        self.k = 8
        
        xyz = xyz.transpose(1,2)
        feat = feat.transpose(1,2)
        newxyz,groupfeat = pointnet2_utils.sample_and_groupfeat(self.out_point,self.radius,self.k,xyz,feat)
        xyz = xyz.transpose(1,2)
        feat = feat.transpose(1,2)
        #print(groupfeat.shape,groupfeat.permute(1,2,3,0).shape,self.in_channel)
        print(groupfeat.shape)
        groupfeat = groupfeat.transpose(2,3).view(-1,F,self.k)
        print(groupfeat.shape)
        groupfeat = self.conv(groupfeat).max(dim=-1)[0].view(B,self.out_point,self.out_channel)
                        
        newfeat = groupfeat
        
        return newxyz,newfeat
    
class downWalkBlock(nn.Module):
    
    def __init__(self,in_point,out_point,in_channel,out_channel, radius):
        
        #input: point B,3,N feat: B,C,N
        #output: point B,3,N' feat: B,C',N'
        
        super().__init__()
        
        self.in_point = in_point
        self.out_point = out_point
        self.in_channel = in_channel
        self.out_channel = out_channel

        #RESIDUAL BOTTLENECK
        
        bnsize = int(np.ceil(in_channel/2))
        
        self.drop = nn.Dropout(0.2)
        
        self.rl = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(bnsize)
        self.bn2 = nn.BatchNorm1d(in_channel)
        self.bn3 = nn.BatchNorm1d(out_channel)
            
        self.fc1 = nn.Linear(in_channel,bnsize) #bottleneck MLP
        #self.walk1 = AffineGeometry(in_point,8,bnsize)
        self.walk2 = AffineGeometry(in_point,8,bnsize)
        self.fc2 = nn.Linear(bnsize,in_channel)
        
        #DOWNSAMPLE
        
        self.downsample = downsampleBlock(in_point,out_point,in_channel,out_channel,radius)
        self.fc3 = nn.Linear(out_channel,out_channel)
        
    def forward(self,xyz,feat):
                
        x = self.fc1(feat.transpose(1,2))
        #print(x.shape)
        x = self.rl(self.bn1(x.transpose(1,2)))
        #x = self.walk1(xyz,x)
        x = self.walk2(xyz,x)
        x = self.fc2(x.transpose(1,2))
        x = self.rl(self.bn2(x.transpose(1,2)))
        
        x = x*feat
        
        xyz,x = self.downsample(xyz,x)
        x = self.fc3(x).transpose(1,2)
        x = self.rl(self.bn3(x))
        
        return xyz,x

class upWalkBlock(nn.Module):
    
    def __init__(self,in_point,out_point,in_channel,out_channel):
        
        super().__init__()
        
        self.in_point = in_point
        self.out_point = out_point
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.rl = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channel)
        
        self.upblock = pointMLP.PointNetFeaturePropagation(out_channel*3,out_channel)
        self.walk1 = AffineGeometry(out_point,8,out_channel)
        
        self.fc = nn.Linear(out_channel,out_channel)
    
        
    def forward(self,xyz,xyz2,point1,point2):
        
        feat = self.upblock(xyz.transpose(1,2),xyz2.transpose(1,2),point1,point2)
        
        feat = self.walk1(xyz,feat)
        
        feat = self.rl(self.bn(self.fc(feat.transpose(1,2)).transpose(1,2)))
        
        return feat
    
class WalkSeg(nn.Module):
    
    def __init__(self,num_class):
        super().__init__()
        
        self.pointSize = ((600,300),(300,150),(150,75),(75,30))
        self.featSize = ((10,20),(20,40),(40,80),(80,160))
        self.radii = (0.1,0.2,0.4,0.6)
        
        self.blocks = 12
        
        self.walk1 = AffineGeometry(600,8,3)
        
        self.downblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        
        for i in range(self.blocks):
            j = self.blocks-i-1
            self.downblocks.append(downWalkBlock(self.pointSize[i][0],self.pointSize[i][1],self.featSize[i][0],self.featSize[i][1],self.radii[i]))
            self.upblocks.append(upWalkBlock(self.pointSize[j][1],self.pointSize[j][0],self.featSize[j][1],self.featSize[j][0]))
            
        self.fc0 = nn.Linear(3,10)
        self.fc1 = nn.Linear(10,16)
        self.fc2 = nn.Linear(16,num_class)
        
        self.bn = nn.BatchNorm1d(3)
        
    def forward(self,xyz):
        
        #print(xyz.shape)
        
        feat = self.walk1(xyz,xyz)
        feat = self.fc0(feat.transpose(1,2)).transpose(1,2)
        
        ifeat = feat
        
        featlist = []
        xyzlist = []
        
        for i in range(self.blocks):
        
            #print(xyz.shape,feat.shape)
            
            featlist.append(feat)
            xyzlist.append(xyz)
            xyz,feat = self.downblocks[i](xyz,feat)
            xyz = xyz.transpose(1,2)
            
        for i in range(self.blocks):
            
            j = self.blocks-i-1
            #print("STUFF:",j,featlist[j].shape,feat.shape)
            feat = self.upblocks[i](xyzlist[j],xyz,featlist[j],feat)
            xyz = xyzlist[j]
            
        #print(feat[0],ifeat[0])
        
        feat = self.fc1(feat.transpose(1,2))
        feat = self.bn(self.fc2(feat).transpose(1,2))
        
        #print(feat.shape)
        
        feat = F.log_softmax(feat,dim=1)
        
        return feat,None
    
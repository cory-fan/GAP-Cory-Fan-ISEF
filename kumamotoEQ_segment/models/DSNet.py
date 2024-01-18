
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
import models.pointnet2_utils as pointnet2_utils

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

class GeomWalk(nn.Module):
    
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
        self.batchNorm = nn.BatchNorm1d(n_points)
        self.rl = nn.ReLU()
        
    def forward(self,xyz, feat): #(B,N,C) to #(B,N,C'), where C is point dimensionality and C' is geometric features
        
        #prescale = feat.detach().sum(dim=1,keepdim=True)
        B, C, N = xyz.shape
        _, F, _ = feat.shape
        
        #print(xyz.shape,feat.shape)
        affineFeat = self.getLaplacian(xyz,self.k,feat)
        #print(affineFeat.shape)
        
        affineFeat = self.affine(affineFeat)
    
        #affineFeat = affineFeat.mean(dim=0)
        #print(xyz.shape,feat.shape,affineFeat.shape)
        affineFeat = affineFeat.view((B,N,F))
        #postscale = feat.sum(dim=1,keepdim=True)
        #print(prescale.shape,postscale.shape,affineFeat.shape)
        #affineFeat = affineFeat*(prescale/postscale).transpose(1,2)
        #affineFeat = self.transform(affineFeat)
        #affineFeat = self.rl(self.batchNorm(affineFeat))
        
        return affineFeat.transpose(1,2)
    
class FeatureLaplacian(nn.Module):
    
    def getLaplacian(self,xyz, k, feat):
        
        _,F,_ = feat.shape
        
        xyz = xyz.transpose(1,2)
                
        kNeighbors = bknn(xyz[:,0:2,:],k)
        
        allLaplacians = []
                
        for batchn in range(kNeighbors.shape[0]):
            iKNeighbors = kNeighbors[batchn]
            localfeat = feat[batchn,:,torch.flatten(iKNeighbors)]
            localfeat = localfeat.view(iKNeighbors.shape[0],-1,F)
            #print(feat[batchn].shape,localfeat.mean(dim=1).shape)
            allLaplacians.append(feat[batchn].transpose(0,1)-localfeat.mean(dim=1))
            
        resfeat = torch.stack(allLaplacians).transpose(1,2)
        
        return resfeat
        
    def __init__(self,n_points, k_value, n_feat):
        super().__init__()
                
        self.k = k_value
        self.npoints = n_points
                                 
        self.nfeat = n_feat #C'
        
        self.transform = nn.Linear(self.nfeat,self.nfeat) #(B,N,C') to (B, C', N)
        self.batchNorm = nn.BatchNorm1d(n_feat)
        self.rl = nn.ReLU()
        
    def forward(self,xyz, feat): #(B,N,C) to #(B,N,C'), where C is point dimensionality and C' is geometric features
        B, N, C = xyz.shape
        _, _, F = feat.shape
        
        lapfeat = self.getLaplacian(xyz,self.k,feat)
        
        trans = self.transform(lapfeat.transpose(1,2))
        
        trans = self.batchNorm(trans.transpose(1,2)).transpose(1,2)
        trans = self.rl(trans)
        trans = trans.transpose(1,2)
                
        output = feat+trans
        return output
    
class DSgroupMLP(nn.Module):
    
    def getLaplacian(self,xyz, k, feat):
        
        B,F,N = feat.shape
        
        kNeighbors = bknn(xyz[:,0:2,:],k)
        
        allLaplacians = []
        
        for batchn in range(kNeighbors.shape[0]):
            iKNeighbors = kNeighbors[batchn]
            localfeat = feat[batchn,:,torch.flatten(iKNeighbors)]
            localfeat = localfeat.view(F,-1,iKNeighbors.shape[0])
            #localfeat = localfeat.view(N,F,k,)
            #print(feat[batchn].shape,localfeat.mean(dim=1).shape)
            #print(localfeat.shape)
            #print(localfeat.shape)
            allLaplacians.append(localfeat)
        return torch.stack(allLaplacians)
    
    def __init__(self,n_points,n_feat,ksize):
        
        super().__init__()
        
        self.npoints = n_points
        self.nfeat = n_feat
        self.kval = ksize
        
        self.rl = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(n_feat)
        self.fc1 = nn.Linear(n_feat,n_feat)
        self.mp = nn.MaxPool1d(ksize)
        
    def forward(self,xyz,feat):
        
        #print(feat.shape)
        
        B,F,N = feat.shape
        K = self.kval
    
        groupfeat = self.getLaplacian(xyz,self.kval,feat).view(B,F,K,N).transpose(1,3)
        
        x = self.fc1(groupfeat).view(B,N*K,F)
        x = self.rl(self.bn1(x.transpose(1,2))).view(B,F,N,K).max(dim=-1)[0]
        #self.drop(x)
        #x = self.rl(self.bn2(self.fc3(x.transpose(1,2)).transpose(1,2)))
        
        return x
    

    
class resBlock(nn.Module):
    
    def __init__(self,n_points, n_feat):
        
        super().__init__()
        self.nfeat = n_feat
        self.mlp = groupMLP(n_points,n_feat,32)
        self.fc = nn.Linear(n_feat*3,n_feat)
        self.rl = nn.ReLU()
        #self.AG1 = AffineGeometry(n_points,8,n_feat)
        self.AG1 = GeomWalk(n_points,min(n_points,32),n_feat)
        self.AG2 = GeomWalk(n_points,min(n_points,32),n_feat)
        self.AG3 = GeomWalk(n_points,min(n_points,32),n_feat)

        self.bn = nn.BatchNorm1d(n_feat)
        
        
    def forward(self,xyz,feat):
        
        B,N,_ = xyz.shape
        #xyz = func.sigmoid(self.spatial(xyz.transpose(1,2))).transpose(1,2)
        xyz = xyz.transpose(1,2)
        mlpfeat = self.mlp(xyz,feat)
        
        #print("FEAT:",mlpfeat.shape)
        
        feat1 = self.AG1(xyz,mlpfeat)
        feat2 = self.AG2(xyz,feat1)
        feat3 = self.AG3(xyz,feat2)
        
        #print(feat[0,0,0,128:131],xyz[0,:,0])
        
        #print("FFFEAT:",feat.shape,mlpfeat.shape,smallFeat.shape,largeFeat.shape,self.nfeat)
        newfeat = torch.concat((feat1,feat2,feat3),dim=1).transpose(1,2)
        
        x = self.fc(newfeat)
        #endfeat = self.rl(self.bn(x.transpose(1,2)))
            
        #print(mlpfeat.shape,endfeat.shape,self.nfeat)
        #newfeat = self.bn(newfeat.transpose(1,2))
        
        return mlpfeat+self.rl(self.bn(x.transpose(1,2)))

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'leakyrelu0.2':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        return nn.ReLU(inplace=True)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize=None, **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel=3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        # fps_idx = farthest_point_sample(xyz, self.groups).long()
        fps_idx = pointnet2_utils.farthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz],dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]	
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        out_channels = int(out_channels)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        x = self.net(x)
        return x


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, npoint, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3+2*channels if use_xyz else 2*channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        
        #print(channels)
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(PointNetFeaturePropagation, self).__init__()
        self.fuse = ConvBNReLU1D(in_channel, out_channel, 1, bias=bias)
        self.extraction = PosExtraction(out_channel, blocks, groups=groups,
                                        res_expansion=res_expansion, bias=bias, activation=activation)
        self.inc = in_channel


    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
            
        new_points = new_points.permute(0, 2, 1)
        new_points = self.fuse(new_points)
        new_points = self.extraction(new_points)
        return new_points
    
class LUMLP(nn.Module):
    
    def __init__(self,npoints,nfeat,k):
        super().__init__()
        redfeat = int(nfeat/2)
        self.mlp1 = ConvBNReLU1D(nfeat,redfeat)
        self.gmlp = DSgroupMLP(npoints,redfeat,8)
        self.lu = FeatureLaplacian(npoints,k,redfeat)
        self.mlp2 = ConvBNReLU1D(redfeat,nfeat)
        self.mlp3 = ConvBNReLU1D(nfeat,nfeat*2)
        
    def forward(self,xyz,feat):
        x = self.mlp1(feat)
        x = self.gmlp(xyz,x)
        xyz = xyz.transpose(1,2)
        x = self.lu(xyz,x)
        x = self.mlp3(self.mlp2(x)+feat)
        
        return x
    
class LUUpSample(nn.Module):
    
    def __init__(self,npoint,nfeat,k):
        super().__init__()
        self.upsample = PointNetFeaturePropagation(nfeat*3,nfeat)
        self.lu = FeatureLaplacian(npoint,k,nfeat)
    
    def forward(self,xyz1,xyz2,feat1,feat2):
        x = self.upsample(xyz1,xyz2,feat1,feat2)
        x = self.lu(xyz1,x)
        return x
    
def fps(pc,feat):
    
    B,N,_ = pc.shape
    
    idx = pointnet2_utils.farthest_point_sample(pc,int(N/2))
    
    downsampled = []
    downfeat = []
    for i in range(B):
        downsampled.append(pc[i][idx[i]])
        downfeat.append(feat[i][idx[i]])
        
    return torch.stack(downsampled),torch.stack(downfeat)

class DSNet(nn.Module):
    
    def __init__(self,num_class=3,depth=3):
        super().__init__()
        
        self.npoint = 600
        self.nfeat = 128
        self.depth = depth
        
        self.featExpand = nn.Linear(3,128)
        self.lumlp = LUMLP(600,128,8)
        self.encoderlist = nn.ModuleList()
        self.decoderlist = nn.ModuleList()
        pointcount = self.npoint
        featcount = self.nfeat
        for i in range(depth):
            pointcount/=2
            pointcount = int(pointcount)
            featcount*=2
            self.encoderlist.append(LUMLP(pointcount,featcount,8))
            
        for i in range(depth):
            self.decoderlist.append(LUUpSample(pointcount,featcount,8))
            pointcount*=2
            featcount/=2
            featcount = int(featcount)
                    
        self.classifier = nn.Linear(256,3)
        
    def forward(self,xyz):
        x = xyz
        x = self.featExpand(x.transpose(1,2)).transpose(1,2)
        x = self.lumlp(xyz,x)
        x_list = []
        xyz_list = []
        for i in range(self.depth):
            x_list.append(x)
            xyz_list.append(xyz)
            xyz,x = fps(xyz.transpose(1,2),x.transpose(1,2))
            xyz = xyz.transpose(1,2)
            x = x.transpose(1,2)
            x = self.encoderlist[i](xyz,x)
            
        x_list = x_list
        
        for i in range(self.depth):
            idx = self.depth-1-i
            x = self.decoderlist[i](xyz_list[idx].transpose(1,2),xyz.transpose(1,2),x_list[idx],x)
            xyz = xyz_list[idx]
            
        x = x.transpose(1,2)
        x = self.classifier(x)
        
        x = F.log_softmax(x)
        
        return x,None
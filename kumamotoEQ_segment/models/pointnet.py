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

class GeometricUnit(nn.Module):
    
    def getLaplacian(self,xyz, k):
        
        kNeighbors = bknn(xyz[:,0:2,:],k)
        
        allLaplacians = []
        
        for batchn in range(kNeighbors.shape[0]):
            iKNeighbors = kNeighbors[batchn]
            localXYZ = xyz[batchn,:,torch.flatten(iKNeighbors)]
            localXYZ = localXYZ.view(iKNeighbors.shape[0],-1,3)
            allLaplacians.append(xyz[batchn].transpose(0,1)-localXYZ.mean(dim=1))
        return torch.stack(allLaplacians)
    
    def getNormals(self,xyz,k):
        
        kNeighbors = bknn(xyz[:,0:2,:],k)
        
        allNormals = []
        
        for batchn in range(kNeighbors.shape[0]):
            iKNN = kNeighbors[batchn]            
            localXYZ = xyz[batchn,:,torch.flatten(iKNN)]
            localXYZ = localXYZ.view(iKNN.shape[0],-1,3)
            try:
                sol = la.lstsq(localXYZ,torch.ones((localXYZ.shape[0],localXYZ.shape[1],1)).cuda()) #linear regression
                normalvec = sol[0].view((sol[0].shape[0],3))
                #print(normalvec.shape,torch.sum(normalvec*normalvec,1).shape)
                nuvec = normalvec.transpose(0,1)/torch.sqrt(torch.sum(normalvec*normalvec,1))
            except: #unsolvable plane
                nuvec = torch.zeros(xyz[batchn].shape).cuda()

            allNormals.append(nuvec.transpose(0,1))
            
        return torch.stack(allNormals)
        
        
    def __init__(self,n_points,featureType):
        super().__init__()
        
        self.featureType = featureType
        
        self.k = 4
        self.npoints = n_points
                                 
        self.nfeat = 3 #C'
        
        self.transform = nn.Linear(self.nfeat,self.nfeat) #(B,N,C') to (B, C', N)
        self.batchNorm = nn.BatchNorm1d(n_points)
        self.rl = nn.LeakyReLU(0.1)
        
    def forward(self,xyz): #(B,N,C) to #(B,N,C'), where C is point dimensionality and C' is geometric features
        B, N, C = xyz.shape
            
        if (self.featureType == "Laplacian"):
            feat = self.getLaplacian(xyz,self.k)
        if (self.featureType == "Normal"):
            feat = self.getNormals(xyz,self.k)
        
        #print(feat.shape,B,N,C)
        #print(laplacians.shape,normals.shape)
        #allfeat = torch.concat((laplacians,normals),dim=-1)
        #print(allfeat.shape)
        trans = self.transform(feat)

        trans = self.batchNorm(trans)
        trans = self.rl(trans)
        trans = trans.transpose(1,2)
        
        output = xyz+trans
        return output
    
class FeatureLaplacian(nn.Module):
    
    def getLaplacian(self,xyz, k, feat):
        
        _,F,_ = feat.shape
        
        kNeighbors = bknn(xyz[:,0:2,:],k)
        
        allLaplacians = []
        
        for batchn in range(kNeighbors.shape[0]):
            iKNeighbors = kNeighbors[batchn]
            localfeat = feat[batchn,:,torch.flatten(iKNeighbors)]
            localfeat = localfeat.view(iKNeighbors.shape[0],-1,F)
            #print(feat[batchn].shape,localfeat.mean(dim=1).shape)
            allLaplacians.append(feat[batchn].transpose(0,1)-localfeat.mean(dim=1))
        return torch.stack(allLaplacians)
        
    def __init__(self,n_points, k_value, n_feat):
        super().__init__()
                
        self.k = k_value
        self.npoints = n_points
                                 
        self.nfeat = n_feat #C'
        
        self.transform = nn.Linear(self.nfeat,self.nfeat) #(B,N,C') to (B, C', N)
        self.batchNorm = nn.BatchNorm1d(n_points)
        self.rl = nn.ReLU()
        
    def forward(self,xyz, feat): #(B,N,C) to #(B,N,C'), where C is point dimensionality and C' is geometric features
        B, N, C = xyz.shape
        _, _, F = feat.shape
        
        lapfeat = self.getLaplacian(xyz,self.k,feat)
        
        trans = self.transform(lapfeat)

        trans = self.batchNorm(trans)
        trans = self.rl(trans)
        trans = trans.transpose(1,2)
        
        output = feat+trans
        return output
    
class LaplacianExtractor(nn.Module):
    
    def __init__(self,nfeat,npoints,ngeomfeat):
        super().__init__()
        self.laplacianUnit = GeometricUnit(npoints,"Laplacian")
        self.normalUnit = GeometricUnit(npoints,"Normal")
                                                            
        self.bn = nn.BatchNorm1d(nfeat)
        #self.fc = nn.Linear(ngeomfeat,nfeat)
        self.rl = nn.LeakyReLU(0.1)
        self.attn = nn.MultiheadAttention(nfeat,4,kdim=3,vdim=nfeat+3)        

        #self.rl = nn.ReLU()
        
    def forward(self,xyz,feat):
        #print(xyz.shape,feat.shape)
        lapfeat = self.laplacianUnit(xyz).transpose(1,2)
        #normfeat = self.normalUnit(xyz.transpose(1,2)).transpose(1,2)
        #allfeat = torch.concat((lapfeat,normfeat),dim=-1)
        allfeat = lapfeat
        query = feat
        key = lapfeat
        value = torch.concat((feat,lapfeat),dim=-1)
        allfeat = self.fc(allfeat).transpose(1,2)
        allfeat = self.rl(self.bn(allfeat))
        #allfeat = self.attn(query,key,value,need_weights=False)[0].transpose(1,2)
        #allfeat = self.rl(self.bn(allfeat))
        lgfeat = allfeat+feat
        #feat = feat.transpose(1,2)
        #llfeat = self.attn(feat,lapfeat,lapfeat,need_weights=False)[0].transpose(1,2)
        #print(llfeat.shape)
        #llfeat = self.bn(llfeat)
        #print((llfeat-feat).mean())
        
        return lgfeat

class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        
        self.lp1 = FeatureLaplacian(600, 4, 3)
        self.lp2 = FeatureLaplacian(512, 4, 96)
        self.lp3 = FeatureLaplacian(256, 4, 256)
        self.lp4 = FeatureLaplacian(64, 3, 512)
        #self.lpe1 = LaplacianExtractor(96,512,3)
        #self.lpe2 = LaplacianExtractor(256,256,3)

        self.sa1 = PointNetSetAbstractionMsg(512, [0.05, 0.1], [16, 32], 3, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        
        norm = self.lp1(xyz, xyz)

        l0_points = norm
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        #print("1:",l1_xyz.shape,l1_points.shape)
        l1_points = self.lp2(l1_xyz, l1_points)
        #print(l1_points.shape)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        #print("2:",l2_xyz.shape,l2_points.shape)
        l2_points = self.lp3(l2_xyz, l2_points)
        #print(l2_points.shape)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        #l3_points = self.lp4(l3_xyz,l3_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        #print(l4_xyz.shape)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        #print(x.shape)
        return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight=torch.tensor([0.5,1.4,3.4]).cuda()): #[0.5,1.4,3.43]
        
        #pred = pred.view(-1,3)
        #target = target.view(-1)
        #if (weight==None):
        #    total_loss = F.nll_loss(pred, target)
        #else:
        total_loss = F.nll_loss(pred, target, weight=weight)
        return total_loss
    
# class LUMLP(nn.Module):
    
#     def __init__(self):
#         super().__init__()
        
    
# class DSNet(nn.Module):
    
#     def __init__(self):
        

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 3, 600)
    (model(xyz))
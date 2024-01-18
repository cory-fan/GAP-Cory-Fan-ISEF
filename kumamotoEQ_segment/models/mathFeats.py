import torch
import math
from models.pointnet2_utils import farthest_point_sample,index_points

def bknn(x, k):
    inner = 2*torch.bmm(x.transpose(2,1), x)
    xx = torch.sum(torch.square(x), dim=1, keepdim = True)
    pairwise_dist = xx - inner + xx.transpose(2, 1)
    inv_dist = (-1)*pairwise_dist
    
    idx = inv_dist.topk(k=k,dim=-1)
    
    return idx[1]

def getNormals(xyz,k):

    kNeighbors = bknn(xyz[:,0:3,:],k)

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

def getPlanars(xyz,k):
    B,_,N = xyz.shape
    xyz = xyz[:,0:3,:]
    idx = bknn(xyz,k)
    gxyz = index_points(xyz.transpose(1,2),idx.view(B,-1)).view(B,N,k,-1)
    diff = gxyz-xyz.transpose(1,2).unsqueeze(2)
    norms = getNormals(xyz,k)
    outplanar = torch.sum(norms.unsqueeze(2).repeat(1,1,k,1)*diff,dim=-1,keepdim=True)
    inplanar = torch.sqrt(torch.sum(diff*diff,dim=-1,keepdim=True)-torch.sum(outplanar*outplanar,dim=-1,keepdim=True))
    return torch.concat((outplanar,inplanar),dim=-1).transpose(1,3)

def deltaRFunc(t,const,theta): #approximating derivatives below!
    BN,_,_ = const.shape
    ival = torch.tensor(math.cos(theta)).cuda()
    jval = torch.tensor(math.sin(theta)).cuda()
    const = const.squeeze()
    kval = 2*const[:,0]*t*(math.cos(theta)**2)+2*t*const[:,1]*(math.sin(theta)**2)+2*t*const[:,2]*(math.cos(theta)*math.sin(theta))+const[:,3]*(math.cos(theta))+const[:,4]*(math.sin(theta))
    return torch.concat((ival.repeat((BN)).view(-1,1),jval.repeat((BN)).view(-1,1),kval.view(-1,1)),dim=1)

def TFunc(t,const,theta):
    deltaR = deltaRFunc(t,const,theta)
    magDeltaR = torch.sqrt(torch.sum(deltaR*deltaR,dim=1,keepdim=True))
    return deltaR/magDeltaR    

def getCurvature(theta,const):
    deltaT = (TFunc(0.0001,const,theta)-TFunc(0,const,theta))/(0.0001) #derivative approximation
    magDeltaT = torch.sqrt(torch.sum(deltaT*deltaT,dim=1,keepdim=True))
    deltaR = deltaRFunc(0,const,theta)
    magDeltaR = torch.sqrt(torch.sum(deltaR*deltaR,dim=1,keepdim=True))
    kval = magDeltaT/magDeltaR
    return kval

def getPrincipals(const):
    allK = torch.concat([getCurvature(6.28/50*i,const) for i in range(50)],dim=1)
    minK = torch.min(allK,dim=1,keepdim=True)[0]
    maxK = torch.max(allK,dim=1,keepdim=True)[0]
    return minK,maxK

def shapeIndex(const):
    minK,maxK = getPrincipals(const)
    shapeIDX = 1/2-(torch.atan((maxK+minK)/(maxK-minK+0.000001))/math.pi)
    return shapeIDX
    
def regressQuadratic(xyz):
    B,_,N = xyz.shape
    #print(B,N)
    k = min(32,N)
    idx = bknn(xyz,k)
    gxyz = index_points(xyz.transpose(1,2),idx.view(B,-1)).view(B,N,-1).view(B*N,k,-1)
    #print(gxyz.shape)
    x = gxyz[:,:,0].unsqueeze(2)
    y = gxyz[:,:,1].unsqueeze(2)
    z = gxyz[:,:,2].unsqueeze(2)
    feat = torch.concat([x*x,y*y,x*y,x,y,torch.ones(x.shape).cuda()],dim=-1)
    #print("FZ:",feat.shape,z.shape)
    const = torch.linalg.lstsq(feat,z)[0]
    return const

def getLaplacian(xyz,k):
    B,N,_ = xyz.shape
    idx = bknn(xyz.transpose(1,2),k).view(B,N*k)
    groups = index_points(xyz,idx).view(B,k,N,3).transpose(1,2)
    laplacians = xyz-groups.mean(dim=-2)
    return xyz
    
def getfeat(xyz):
    B,N,_ = xyz.shape
    normals = getNormals(xyz.transpose(1,2),16).transpose(1,2)
    const = regressQuadratic(xyz.transpose(1,2))
    shapeIDX = shapeIndex(const).view(B,1,N)
    laplacians = getLaplacian(xyz,16).transpose(1,2)
    feat = torch.concat([xyz.transpose(1,2),normals,shapeIDX,laplacians],dim=1)
    return feat.transpose(1,2)
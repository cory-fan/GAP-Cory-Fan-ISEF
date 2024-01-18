from dataloader import SegData
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import time

from models.pointMLP import pointMLP
from models.lapMLP import lapMLP
from models.DSNet import DSNet
from models.GAPGCN import GAPGCN
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation,farthest_point_sample, index_points

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_count = 3

makeLog = False
    
def euclidean(tensor1, tensor2):
    selfdots1 = torch.tensor([torch.dot(i,i) for i in tensor1]).cuda()
    selfdots2 = torch.tensor([torch.dot(i,i) for i in tensor2]).cuda()
    #rint(tensor1,tensor2)
    mm = torch.matmul(tensor1,tensor2.transpose(0,1))
    sd1 = (selfdots1*torch.ones((mm.shape[1],mm.shape[0])).cuda()).transpose(0,1)
    sd2 = (selfdots2*torch.ones((mm.shape[0],mm.shape[1])).cuda())
    #print(sd1.device,sd2.device)
    return (sd1+sd2-2*mm)

def interpolate(pointset,target,npoints):
    fullpoints = []
    oset = (torch.zeros(pointset.shape[1],device=pointset.device)+1.0)
    for i,points in enumerate(pointset):
        #print(np.logical_not((np.abs(points[:,0])==oset)*(np.abs(points[:,1])==oset)*(np.abs(points[:,2])==oset)))
        cpoints = points[torch.logical_not((torch.abs(points[:,0])==oset)*(torch.abs(points[:,1])==oset)*(torch.abs(points[:,2])==oset))].cuda()
        x = torch.rand((npoints-len(cpoints)))*2.0-1.0
        y = torch.rand((npoints-len(cpoints)))*2.0-1.0
        xy = torch.stack([x,y]).to(cpoints.device)
        #print(xy.device,cpoints.device)
        #print(xy.device,cpoints.device)
        dist = euclidean(xy.transpose(0,1),cpoints[:,0:2].float())
        ret = torch.topk(dist,10)
        val = ret[0]
        idx = ret[1]

        newz = []
        newtargets = []
        for j,i in enumerate(val):
            #print(i.shape,idx.shape)
            base = torch.sum(i**(-1))
            top = torch.sum((i**(-1)*cpoints[idx[j],2]))
            z = top/base
            #print(base,top,cpoints[idx[j],2])\
            #print(z)
            newtargets.append(target[idx[0]])
            newz.append(z)

        newpoints = torch.stack([x,y,torch.tensor(newz,device=x.device)]).to(cpoints.device).transpose(0,1)
        #print(newpoints.shape)
        augpoints = torch.concat([cpoints,newpoints])
        fullpoints.append(augpoints)
    return torch.stack(fullpoints)

def truncate(points,target):
    B,N,C = points.shape
    cond = torch.max(torch.abs(torch.tensor(points)),dim=-1)[0]<1
    points = torch.where(cond.unsqueeze(2),torch.tensor(points),torch.ones((B,N,C)))
    targets = torch.where(cond,torch.tensor(target),torch.zeros((B,N)))
            
    return points,target
        

def augdata(points,target,hard=True):
    points = points.cpu().numpy()
    if (hard):
        points[:, :, 0:3] = provider.jitter_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3], shift_range=0.4)
        points,target,_ = provider.shuffle_data(points,target)
    points,target = truncate(points,target)
    points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3], shift_range=0.2)
    points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
    #points = interpolate(points,target,600)
    points = torch.tensor(points).float()
    #points = interpolate(points,720)
    return points,target

def parse_args():
    """ PARAMETERS """
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training [default: 24]')
    parser.add_argument('--model', default='lapnet', help='model name [default: POINTNET3_MSG_CLS]')
    parser.add_argument('--modelnum', default=5, help='model name [default: POINTNET3_MSG_CLS]')
    parser.add_argument('--epoch', default=1600, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.0003, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--ngpu', type=int, default=[0, 1, 2], help='specify how many gpus to train the model [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores with voting [default: 3]')
    parser.add_argument('--test_path', default='./data/pointSeg/test', help='path of the test data')
    parser.add_argument('--train_path', default='./data/pointSeg/train', help='path of the train data')
    parser.add_argument("--worker", type=int, default=torch.get_num_threads())
    return parser.parse_args()


def test(model, loader, num_class=class_count, vote_num=1):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    accMat = np.zeros((num_class,2,2))
    for points, target in tqdm(loader):
        points,target = augdata(points,target,hard=False)
        points = points.transpose(2, 1)
        points, target = points.float().to(device), target.to(device)
        classifier = model.eval()
        #vote_pool = torch.zeros(target.shape[0], num_class).to(device)
        #for _ in range(vote_num):
        pred, _ = classifier(points)
            #vote_pool += pred
        #pred = vote_pool / vote_num
        pred = pred.reshape((-1,3))
        target = target.view(-1)
        pred_choice = pred.data.max(1)[1]
        
        for cat in torch.unique(target):
            c = int(cat.item())
            points = points.cuda()
            target = target.cuda()
            classacc = pred_choice[target == cat].eq(target[target == cat]).cpu().sum()
            classacc = classacc.cpu()
                        
            truePos = classacc #true pos
            falsePos = (pred_choice==cat).float().sum()-classacc
            falseNeg = (target==cat).float().sum()-classacc
            trueNeg = len(pred_choice)-truePos-falsePos-falseNeg
            
            accMat[c,0,0]+=falseNeg
            accMat[c,1,0]+=trueNeg
            accMat[c,0,1]+=falsePos
            accMat[c,1,1]+=truePos
            
            points = points.cpu()
            target = target.cpu()
            #print(classacc,len(pred_choice[target==c]),classacc.item() / (target==c).float().sum())
            class_acc[c, 0] += classacc.item() / (target==c).float().sum()
            class_acc[c, 1] += 1
        pred_choice = pred_choice.cpu()
        correct = pred_choice.eq(target).cpu().sum()
        batch_num_samples = points.size()[0]
        mean_correct.append(correct.item() / (batch_num_samples*600))
    precision = np.zeros(num_class)
    recall = np.zeros(num_class)
    f1 = np.zeros(num_class)
    
    print(accMat[0,1,1]+accMat[0,0,0],accMat[1,1,1]+accMat[1,0,0],accMat[2,1,1]+accMat[2,0,0])
    
    for i in range(num_class):
        precision[i] = accMat[i,1,1]/(accMat[i,1,1]+accMat[i,0,1])
        recall[i] = accMat[i,1,1]/(accMat[i,1,1]+accMat[i,0,0])
        f1[i] = 2*precision[i]*recall[i]/(precision[i]+recall[i])
        
    totalprecision = precision.mean()
    totalrecall = recall.mean()
    totalF1 = 2*totalprecision*totalrecall/(totalprecision+totalrecall)
    print("C:",class_acc[:,0],class_acc[:,1])
    class_acc[:, 2] = np.nan_to_num(class_acc[:, 0] / class_acc[:, 1])
    print("Class_Acc:",class_acc[:,2])
    print("Recall:",recall,totalrecall)
    print("Precision:",precision,totalprecision)
    #print("F1:",f1,totalF1)
    
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc, totalF1


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    """ CREATE DIR """
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    """ LOG """
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    """ DATA LOADING """
    log_string('Load dataset ...')
    TRAIN_DATASET = SegData(args.train_path, npoint=args.num_point)
    TEST_DATASET = SegData(args.test_path, npoint=args.num_point)
    trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testDataLoader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print('num_train_files: ' + str(len(TRAIN_DATASET.filepaths)))
    print('num_val_files: ' + str(len(TEST_DATASET.filepaths)))

   
    """ MODEL LOADING """
    num_class = class_count
    #shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    #shutil.copy('./models/pointnet_util.py', str(experiment_dir))
    #train on the single gpu
    #classifier = MODEL.get_model(num_class).to(device)
    #train on the multiple gpu
    
    MODEL = importlib.import_module(args.model)

    model = int(args.modelnum)
    if (model==0):
        classifier = nn.DataParallel(MODEL.get_model(num_class).to(device))
    elif (model==1):
        classifier = nn.DataParallel(pointMLP(num_class=class_count).to(device))
    elif (model==2):
        classifier = nn.DataParallel(lapMLP(num_class=class_count).to(device))
    elif (model==3):
        pass
    elif (model==4):
        pass
    elif (model==5):
        pass
    elif (model==6):
        pass
    elif (model==7):
        classifier = DSNet().to(device)
    elif (model==8):
        pass
    elif (model==9):
        classifier = GAPGCN().to(device)
    
    criterion = MODEL.get_loss().to(device)
#     checkpoint = torch.load('/home/featurize/work/Cory_Fan/PointView-GCN/PointNet++/log/classification/2023-10-04_02-07/checkpoints/best_model.pth')
#     #start_epoch = checkpoint['epoch']
    
#     stateDict = checkpoint['model_state_dict']
#     newDict = dict()
#     popL = []
#     for i in stateDict.keys():
#         newDict[i.replace('module.','')] = stateDict[i]
        
#     classifier.module.load_state_dict(newDict)
#     #log_string('Use pretrain model')
#     #except:
    log_string('No existing model, starting training from scratch...')
    start_epoch = 0 
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(classifier.parameters(),
                                     lr=args.learning_rate,
                                     betas=(0.995, 0.999),
                                     eps=1e-08, weight_decay=0.0001)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.5, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    mean_correct = []

    """ TRAINING """

    logger.info('Start training...')
    train_instance_acc, instance_acc, class_acc = [], [], []
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        scheduler.step()
        total_loss = 0
        for points, target in tqdm(trainDataLoader):
            points,target = augdata(points,target,hard=True)
            #points = points.numpy()
            #points, target = provider.random_point_dropout(points.clone(), target.clone())
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            #points = torch.tensor(points).float()
            points = points.transpose(2, 1)
            # target = target.unsqueeze(1).repeat(1, 4 * (5)).view(-1)
            points, target = points.to(device), target.long().to(device)
            optimizer.zero_grad()
            batch_num_samples = points.size()[0]
            #stime = time.time()
            classifier = classifier.train()
            pred, trans_feat, = classifier(points.float())
            #print(pred.shape)
            pred = pred.reshape(-1,3)
            target = target.view(-1)
            loss = criterion(pred, target, trans_feat)
            total_loss+=loss.detach()
            pred_choice = pred.data.max(1)[1]
            #print(pred_choice,target)
            #print(pred_choice.shape,target.shape)
            correct = pred_choice.eq(target).cpu().sum()
            correct_ = correct.item() / (batch_num_samples*600)
            mean_correct.append(correct_)
            loss.backward()
            #print(time.time()-stime)
            optimizer.step()
            global_step += 1
        _train_instance_acc = np.mean(mean_correct)
        with torch.no_grad():
            _instance_acc, _class_acc, f1 = test(classifier, testDataLoader)
            if _instance_acc >= best_instance_acc:
                best_instance_acc = _instance_acc
                best_epoch = epoch + 1
            if _class_acc >= best_class_acc:
                best_class_acc = _class_acc
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/class_model.pth'
                log_string('Saving at %s' % savepath)
                state = {'epoch': best_epoch,
                         'instance_acc': _instance_acc,
                         'class_acc': _class_acc,
                         'model_state_dict': classifier.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict()}
                torch.save(state, savepath)
            if _instance_acc >= best_instance_acc:
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {'epoch': best_epoch,
                         'instance_acc': _instance_acc,
                         'class_acc': _class_acc,
                         'model_state_dict': classifier.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict()}
                torch.save(state, savepath)
            # if True:
            #     #logger.info('Save model...')
            #     #savepath = str(checkpoints_dir) + '/last_model.pth'
            #     #log_string('Saving at %s' % savepath)
            #     #state = {'epoch': best_epoch,
            #              'instance_acc': _instance_acc,
            #              'class_acc': _class_acc,
            #              'model_state_dict': classifier.state_dict(),
            #              'optimizer_state_dict': optimizer.state_dict()}
            #     #torch.save(state, savepath)
            global_epoch += 1
        print("LOSS:",total_loss/len(trainDataLoader))
        log_string('Train Instance Accuracy: %f' % _train_instance_acc)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (_instance_acc, _class_acc))
        log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))
        log_string('F1:'+str(f1))
        train_instance_acc.append(_train_instance_acc)
        instance_acc.append(_instance_acc)
        class_acc.append(_class_acc)
    logger.info('End of training...')

    """ PLOTTING """
    plt.plot(train_instance_acc, label='Train Accuracy')
    plt.plot(instance_acc, label='Test Accuracy')
    plt.title('Model Accuracy ({} epochs)'.format(args.epoch))
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)

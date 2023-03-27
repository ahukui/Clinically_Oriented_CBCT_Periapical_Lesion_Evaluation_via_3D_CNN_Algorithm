import torch
from model.PALNet import vesselnet as PALNet
import os 
from dataload.liver_seg_2d_v3 import npyDataSet as npyDataSet_2d_v3
from torch.utils.data import DataLoader
import numpy as np
import random
import argparse
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler
from torch.optim import lr_scheduler
import cv2 
from torch.nn.functional import interpolate, adaptive_max_pool3d, avg_pool3d
from bisect import bisect_right
import yaml
from scipy import ndimage
import copy
from model.criterions import BinaryDiceLoss
from torch import nn
torch.manual_seed(7) # cpu
torch.cuda.manual_seed(7) #gpu
np.random.seed(7) #numpy
random.seed(7) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

dataset_dict = {
    "npyDataSet_2d_v3": npyDataSet_2d_v3,

}

network_dict = {
    "PALNet":PALNet,

}

class WarmUpMultiStepLR(MultiStepLR):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, warmup_iters=500):
        self.warmup_iters = warmup_iters
        super(WarmUpMultiStepLR, self).__init__(optimizer, milestones, gamma, last_epoch)
    
    def get_lr(self):
        if self.last_epoch<self.warmup_iters:
            return [base_lr * 0.1 for base_lr in self.base_lrs]
        return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch) for base_lr in self.base_lrs]

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, last_epoch=-1, warmup_iters=500):
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        super(WarmUpLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch<self.warmup_iters:
            return [base_lr * (0.001+0.999*self.last_epoch/self.warmup_iters) for base_lr in self.base_lrs]
        return [base_lr * (1-self.last_epoch/self.max_iters) for base_lr in self.base_lrs]

class PreFetch():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

def dice_loss(label, score):
    loss = 0
    for index in range(score.shape[1]):
        gt = label.clone()
        gt[gt!=index+1] = 0
        gt[gt>0] = 1
        _score = score[:, index:index+1]
        tp = (_score*gt).sum((1,2,3,4))+1e-5
        loss = loss + 1 - (2*tp/(gt.sum((1,2,3,4))+_score.sum((1,2,3,4))+2e-5)).mean()
        
    return loss



def get_tumor_region(label, score):
    loss = 0
    for batch_idx in range(label.size(0)):
        _label = label[batch_idx].squeeze().cpu().numpy()
        mask = ndimage.label(_label>0)[0]
        #print('mask', np.max(mask))
        if mask.max()<1:
            continue
        for _index in range(1, mask.max()+1):
            z, y, x = np.where(mask==_index)
            loss += dice_loss(label[batch_idx:batch_idx+1, :, z.min()-1:z.max()+2, y.min()-5:y.max()+5, x.min()-5:x.max()+5], 
                              score[batch_idx:batch_idx+1, :, z.min()-1:z.max()+2, y.min()-5:y.max()+5, x.min()-5:x.max()+5])
    return loss

def ce_loss(label, score):
    loss = 0
    scores = []
    for index in range(score.shape[1]):
        gt = label.clone()
        gt[gt!=index+1] = 0
        gt[gt>0] = 1
        _score = score[:, index:index+1]
        pos = _score[gt==1]+1e-5
        neg = 1-_score[gt==0]+1e-5
        neg = neg.sort()[0]
        hard_neg = neg[:pos.numel()*3+1000]
        loss = loss - (pos.log().sum()+neg.log().sum())/gt.numel()
        _score = torch.cat((pos, hard_neg), dim=-1)
        loss = loss - 2.5*((1-_score)**2*_score.log()).mean()
        scores.append(pos)
    scores.append(1.000005-score.max(dim=1, keepdim=True)[0][label==0])
    scores = torch.cat(scores, dim=-1)
    loss = loss - scores.log().sum()/scores.numel()

    return loss



def clip_gradient(optimizer):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    grads = []
    delt = 100
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                grad = ((param.grad.data*param.grad.data).sum())**0.5
                weight = max(((param.data*param.data).sum())**0.5, 0.001)
                if grad/weight>delt:
                    param.grad.data = param.grad.data * delt * weight / grad

def stage_two_train_main(args):
    #dice_loss = BinaryDiceLoss()
    max_iters = 480000
    unet = network_dict[args["network"]](num_seg_classes=args["num_class"], inchannel=args["inchannel"])
    unet = nn.DataParallel(unet)    
    sdf = dataset_dict[args["dataset"]](args["fold"], args["num_image"])
    _data_loader = DataLoader(sdf, batch_size=args["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    model_pth = unet.state_dict()
    if args["resume"]>-1:
        pth = torch.load(os.path.join(args["save_folder"],str(args["resume"]).zfill(5)+"_stage_one.pth"))
        model_pth = pth["model"]
    elif args["pretrain"] is not None: 
        pth = torch.load(args["pretrain"])
        pth = pth["model"]
        for key in pth:
            if key in model_pth:
                if pth[key].shape==model_pth[key].shape:
                    model_pth[key] = pth[key]
                else: 
                    print(key, pth[key].shape, model_pth[key].shape)
    if model_pth is not None:
        print('load pretraining weights')
        unet.load_state_dict(model_pth)
    unet.cuda()
    # solver = torch.optim.SGD(unet.parameters(), lr=args["baselr"], momentum=0.9, weight_decay=5e-5, nesterov=True)
    solver = torch.optim.AdamW(unet.parameters(), eps=1e-8, betas=(0.9, 0.99),
                                lr=1e-3, weight_decay=5e-3)
    lr_schedule = WarmUpLR(solver, max_iters, warmup_iters=5000)
    num_iter = 0
    if args["resume"]>-1:
        lr_schedule.load_state_dict(pth["opti"])
        num_iter = args["resume"]


    unet.train()
     
    
    
    weight = [0.2, 0.4, 0.8]
    flag = True
    while flag:
        data_loader = PreFetch(_data_loader)
        image, sdf = data_loader.next()
        while image is not None:
            solver.zero_grad()
            if random.random()>0.5:
                image = image.flip(3)
                sdf = sdf.flip(3)
            if random.random()>0.5:
                image = image.flip(4)
                sdf = sdf.flip(4)
            if random.random()>0.5:
                image = image.flip(2)
                sdf = sdf.flip(2)
            outputs = unet.forward(image.contiguous())
            del image
            loss = 0
            for index, output in enumerate(outputs):
                if output.shape[-3:]!=sdf.shape[-3:]:
                    _sdf = interpolate(sdf, output.shape[-3:])
                else:
                    _sdf = sdf
                output = output.sigmoid()
                #print(output.shape, _sdf.shape)
                _dice_loss = dice_loss(_sdf[:, :, 1:-1], output[:, :, 1:-1]) + get_tumor_region(_sdf[:, :, 1:-1], output[:, :, 1:-1]) 
                loss += (_dice_loss + ce_loss(_sdf[:, :, 1:-1], output[:, :, 1:-1]))*weight[index]
                #loss += (_dice_loss)*weight[index]
            loss.backward()
            sdf = interpolate(sdf, output.shape[-3:])
            # clip_gradient(solver)
            solver.step()
            output[output>0.5] = 1
            output[output<1] = 0
            liver_tp = (output[sdf==1]).sum()
            liver_dice = (liver_tp*2+1)/(sdf.sum()+output.sum()+1)
            print('----------',num_iter, ', learning rate:', format(lr_schedule.optimizer.param_groups[0]["lr"], ".6f"), ', loss:', format(loss.cpu().item(), ".3f"), 
                        ', liver dice:', format(liver_dice.cpu().item(), ".3f"), 
                        format(output.sum().cpu().item(), ".1f"), format(sdf.sum().cpu().item(), ".1f"))

            del output
            image, sdf = data_loader.next()
            lr_schedule.step()
            if num_iter % 500 == 0 and num_iter>0:
                save_dict = {}
                save_dict["model"] = unet.state_dict()
                save_dict["opti"] = lr_schedule.state_dict()
                torch.save(save_dict, args["save_folder"] + "/" + str(num_iter).zfill(5) + "_stage" + ".pth")
                #print(args["save_folder"] + "/" + str(num_iter).zfill(5) + "__loss" + ".pth")
            if num_iter>=max_iters:
                flag = False
                break
            num_iter += 1

def arg():
    parase = argparse.ArgumentParser()
    parase.add_argument("--config", type=str, required=True)
    return parase.parse_args()

    

if __name__ == "__main__":
    args = arg()
    args = yaml.safe_load(open(args.config, "r"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpu_id"])
    if not os.path.exists(args["save_folder"]):
        os.makedirs(args["save_folder"])
    stage_two_train_main(args)
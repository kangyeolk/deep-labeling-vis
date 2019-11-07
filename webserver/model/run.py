import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Subset

import numpy as np
import time
import datetime
import os
import matplotlib.pyplot as plt
from itertools import combinations

from model import DenseNet
from utils import cal_l2, ContrastiveLoss, sigmoid
from dataloader import TestImageFolder
from matplotlib import colors
import math

# Model Configuration 
batch_size = 16
lr = 1e-5
crop_size = 3*256

# Data path for Training
# t_hp1_dir = '../76_server/test_patch_2304/hyperplastic/1_hyperplastic_10'
# t_nor1_dir = '../76_server/test_patch_2304/normal/1_normal_10'
# t_dir = '../76_server/test_patch_2304/combine' # HP + Normal
# t2_dir = '../76_server/test_patch_2304/combine2' # HP + Normal


# Data dir for Visual Check
# all_hp10_dir = '../212_data/all_1hp10/'
# all_hp20_dir = '../212_data/all_1hp20/'

# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# criterion = nn.CrossEntropyLoss()
# contrasive = ContrastiveLoss(margin=2.0)# .to(device)

# crop_size = 3*256
# transform = transforms.Compose([
#                     transforms.CenterCrop((crop_size, crop_size)),
#                     transforms.Resize((256, 256)),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=(0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])

class Trainer:
    
    def __init__(self, model, minmax_epochs, alpha, batch_size, test_dir, f_lambda=1.0):
        
        # MISC
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prev_val_f_loss = np.inf
        self.curr_val_f_loss = 0
        self.test_dir = test_dir
        self.alpha = alpha
        
        # Archive for saving mean of features
        self.archive = {}
        self.archive['hp'] = {}
        self.archive['nor'] = {}
        self.archive['ta'] = {}
        
        self.archive['hp']['sum'] = np.zeros(224)
        self.archive['hp']['count'] = 0
        # self.archive['hp']['avg'] = 0
        self.archive['nor']['sum'] = np.zeros(224)
        self.archive['nor']['count'] = 0
        # self.archive['nor']['avg'] = 0
        self.archive['ta']['sum'] = np.zeros(224)
        self.archive['ta']['count'] = 0
        # self.archive['ta']['avg'] = 0
        
        
        # Model & Optimizer 
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), betas=(0.5, 0.99))
        self.model = nn.DataParallel(model)
        self.contrasive = ContrastiveLoss(margin=2.0)# .to(device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training configuration
        self.transform = transforms.Compose([
                    transforms.CenterCrop((768, 768)),
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
        self._num_samples = None
        self._data_dir = None
        self.batch_size = batch_size
        self.min_epochs = minmax_epochs[0]
        self.max_epochs = minmax_epochs[1]
        self.batch_size = batch_size
        self.f_lambda = f_lambda

        
    def get_loaders(self):
        """ Prepare data loaders via data directory """
        # It should be replced with procedure where information which have similar index information.. 
        d_set = ImageFolder(root=self._data_dir, transform=self.transform)
        idx = np.random.choice(len(d_set), self._num_samples, replace=False)
        t_idx = np.random.choice(idx, int(self._num_samples*0.8), replace=False)
        v_idx = np.array([xx for xx in idx if xx not in t_idx])
        
        t_subset = Subset(d_set, t_idx)
        v_subset = Subset(d_set, v_idx)
    
        self.t_loader = DataLoader(t_subset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.v_loader = DataLoader(v_subset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        
    
    def train(self):

        print('train()')
        
        # Getting Loaders
        self.get_loaders()
        
        # End flag
        self.end = False
        
        for epoch in range(self.max_epochs):

            print('epoch')

            self.reset_cnts()
        
            ## Train 1-epoch
            self.model.train()
            start_t = time.time()     
            for ii, (image, label) in enumerate(self.t_loader):
#                 print(label)
                image_var = image.to(self.device)
#                 label = self.label.expand(image.size(0)).long()
                label_var = label.to(self.device)

                # Forwarding & Backwarding 
                ff, output = self.model(image_var)
                f = ff.squeeze(-1).squeeze(-1)

                # Feature alignment Loss
                f_loss = 0
                f_dic = {}
                B = image.size(0)
                N = len(f[0])

                for k in sorted(label.unique()):
                    f_dic[k.item()] = f[label == k, :]
                
                # Feature distances Loss / Classification Loss 
                f_loss = self.contrasive(f_dic, B, N)
                cls_loss = self.criterion(output, label_var)

                # Total Loss / Update 
                loss = cls_loss + self.f_lambda * f_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Accuaracy Measure
                pred = output.argmax(dim=-1).cpu()
                self.t_corr_cnt += (pred == label).sum().item()    
                self.t_total_cnt += B

                if (ii+1) % 1 == 0:
                    lap = time.time() - start_t
                    elapsed = str(datetime.timedelta(seconds=lap))
                    print('Elapsed [{}]\t'
                          'Epoch [{}/{}]\t'
                          'Iters [{}/{}]\t\t'
                          'Fet Loss [{:.4f}]\t'
                          'Cls Acu. [{:.4f}]'.format(elapsed, epoch+1, self.max_epochs, (ii+1), len(self.t_loader), 
                                                     f_loss.item(), self.t_corr_cnt/self.t_total_cnt))

                # Free memory
                del f, ff, f_dic, loss, f_loss, cls_loss, output, image_var, label_var
#                 if (ii+1) % 1 == 0:
#                     break
                    
            self.validate(epoch)
            if self.min_epochs > epoch + 1:
                self.end = False
            if self.end and self.min_epochs <= epoch + 1:    
                print('==> Break at {} with F-loss {}'.format(epoch+1, self.curr_val_f_loss))        
                break
        # Return model
        self.return_model()

            
        
    def validate(self, epoch):
        # Validate 1-epoch
        self.model.eval()
        for jj, (image, label) in enumerate(self.v_loader):

            image_var = image.to(self.device)
#             label = self.label.expand(image.size(0)).long()
            label_var = label.to(self.device)

            # Forwarding & Backwarding 
            ff, output = self.model(image_var)
            f = ff.squeeze(-1).squeeze(-1)

            # Feature alignment Loss
            f_loss = 0
            f_dic = {}
            B = image.size(0)
            N = len(f[0])

            for k in sorted(label.unique()):
                f_dic[k.item()] = f[label == k, :]

            # Feature distances Loss / Classification Loss 
            f_loss = self.contrasive(f_dic, B, N)
            cls_loss = self.criterion(output, label_var)

            # Total Loss Measure
            loss = cls_loss + self.f_lambda * f_loss

            # Accuaracy Measure
            pred = output.argmax(dim=-1).cpu()
            self.v_corr_cnt += (pred == label).sum().item()    
            self.v_total_cnt += B
            self.curr_val_f_loss += f_loss.item()

            if (jj+1) % 1 == 0:
                print('#### Validation\t'
                      'Epoch [{}/{}]\t'
                      'Fet Loss [{:.4f}]\t'
                      'Cls Acu. [{:.4f}]'.format(epoch+1, self.max_epochs, 
                                                 f_loss.item(), self.v_corr_cnt/self.v_total_cnt))

            del f, ff, f_dic, loss, f_loss, cls_loss, output, image_var, label_var


        if self.curr_val_f_loss < self.prev_val_f_loss:
            self.prev_val_f_loss = self.curr_val_f_loss
        else:
            self.end = True
    
    def return_model(self):
        self.update_archive()
        return self.model
    
    def update_vecs(self):
        """ Update feature vector with given data for similiarity measurement """

        test_set = ImageFolder(root=self.test_dir, transform=self.transform)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

        print('==> Create feature vector')
        self.feature_vec = {}
        self.feature_vec['temp'] = torch.zeros((0, 100))            
        self.hp_cnt = self.normal_cnt = self.ta_cnt = 0
        
        # Compute feature vectors with test dataset
        for ii, (image, label) in enumerate(self.test_loader):
            image_var = image.to(self.device)
            ft, output = self.model(image_var)
            _ft = ft.squeeze(-1).squeeze(-1)
            
            # Add 
            self.feature_vec['temp'] = torch.cat((self.feature_vec['temp'], _ft.detach().cpu()), dim=0)
                
            # Determine Class
            sx_output = nn.Softmax(dim=-1)(output)
            pred = torch.argmax(sx_output, dim=-1).detach().cpu()
            self.hp_cnt += (pred == 0).sum().item()
            self.normal_cnt += (pred == 1).sum().item()
            self.ta_cnt += (pred == 2).sum().item()
            
            del ft, _ft, sx_output, pred, output
        
        # Compute average feature vector
        curr_key = list(self.feature_vec.keys()) 
        for kk in curr_key:
            if self.feature_vec[kk].nelement() != 0:
                self.feature_vec['%s_avg' % kk] = torch.mean(self.feature_vec[kk], dim=0)
        
        # Select target class via Majority voting
        idx = torch.argmax(torch.Tensor([self.hp_cnt, self.normal_cnt, self.ta_cnt])).item()
        aa = ['hp', 'normal', 'ta']
        self.label = aa[idx]
    
    

        
    def viz_WSI_ft(self, whole_path, whole_wh, alpha, dis_th):
        """
        :whole_path - Directory of file where whole image patchese contain
        :whole_hw   - Height and Width value of whold slide image for confine map sized
        :alpha      - Weight value for adjusting softmax value and distance between feature vector mean and feature
        :label      - Indicator for where to highlight
        :dis_th     - Distance thresholding ==>  Temporary delete
        """
        # 
        self.update_vecs()  
        label_to_id = {'hp':0, 'normal': 1, 'ta': 2}
        # target = label_to_id[self.label]
        target = 2
        
        # Data Path
        w_set = TestImageFolder(root=whole_path, transform=self.transform)
        w_loader = DataLoader(w_set, 
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=4)
        
        # Create empty map
        
        # print(self.map1x1)
        
        
        # Forwarding.. 
        self.model.eval()
        self.map1x1 = np.zeros((whole_wh[1] // 256 +1, whole_wh[0] // 256 +1))
        
        for i, (image, _, image_name) in enumerate(w_loader):

            # import pdb; pdb.set_trace()

            image_var = image.to(self.device)
            feature, confidence = self.model(image_var)
            _feature = feature.squeeze(-1).squeeze(-1)
            
            # Calculate distance for given label 
            _feature = _feature.detach().cpu().numpy()
            # print(self.archive)
            cls_vec = self.feature_vec['temp_avg']
            ex_ft = np.repeat(cls_vec.numpy()[np.newaxis, :], image.size(0), axis=0)
            # print(feature.shape, ex_ft.shape)
            # feature_dis = np.sqrt(np.power(_feature - ex_ft, 2).sum(axis=-1) + 1e-5)
            feature_dis = np.power(_feature - ex_ft, 2).sum(axis=-1)
            feature_dis = (feature_dis - np.min(feature_dis)) / np.max(feature_dis)  
            
            
            # Calculate Total distance ## 1-Distance + Certainty
            # import pdb; pdb.set_trace()    
            total_conf = self.alpha*(1 - sigmoid(feature_dis)) + nn.Softmax(dim=-1)(confidence)[:, target].detach().cpu().numpy()
            
            
            # Fill prediction map in

            for ii in range(image.size(0)):
                # import pdb; pdb.set_trace()
                if total_conf[ii] > dis_th:
                    _image_name = image_name[ii].split(',')
                    # y, x = int(_image_name[-2]), int(_image_name[-1][:-4])
                    y, x = (int(_image_name[-4])//256)+2, (int(_image_name[-3])//256)+2
                    self.map1x1[x, y] = target + 1
                
            # Free memory
            del image_var, feature, _feature, feature_dis, confidence, total_conf

            print("{}/{}".format(i, len(w_loader)))
            
        
        # Print out map
        # Background - Black, HP - Blue, Normal - Green, TA - Red
        cmap = colors.ListedColormap(['k', 'b', 'g','r'])
        # plt.imshow(self.map1x1, cmap=cmap)
        # plt.show()

        return self.map1x1
    
    def sx_validate(self, data_path):
        """ Validate trained model for otehr data """
        
        d_set = ImageFolder(root=data_path, transform=self.transform)
        d_loader = DataLoader(d_set, batch_size=self.batch_size, shuffle=True, num_workers=4)
        
        
        total_cnt = 0
        corr_cnt = 0
        
        self.model.eval()
        for jj, (image, label) in enumerate(d_loader):

            image_var = image.to(self.device)
            label_var = label.to(self.device)

            # Forwarding & Backwarding 
            ff, output = self.model(image_var)

            # Accuaracy Measure
            pred = output.argmax(dim=-1).cpu()
            
            idx = (label != 2) 
            
            corr_cnt += (pred == label)[idx].sum().item()    
            total_cnt += idx.sum().item()
            
            if (jj+1) % 1 == 0:
                print('#### Validation\t'
                      'Cls Acu. [{:.4f}]'.format(corr_cnt/total_cnt))

            del ff, output, image_var, label_var
        
    
    
    @property
    def data_dir(self):
        return self._data_dir
    
    @data_dir.setter
    def data_dir(self, data_dir):
        self._data_dir = data_dir
        
    @property
    def num_samples(self):
        return self._num_samples
    
    @num_samples.setter
    def num_samples(self, num_samples):
        self._num_samples = num_samples
        
    @property
    def curr_label(self):
        return self._curr_label
    
    @curr_label.setter
    def curr_label(self, curr_label):
        self._curr_label = curr_label

# # Model
# model = DenseNet(growthRate=32, depth=24, reduction=0.5, bottleneck=True, nClasses=3)

# trainer = Trainer(model=model, minmax_epochs=(10, 30), batch_size=16, f_lambda=2.0)

# # Train with hp1 200 patches
# trainer.data_dir = t_dir
# trainer.num_samples = 400
# trainer.train()

# trainer.viz_WSI_ft(whole_path=all_hp10_dir, whole_hw=(12955, 77687), alpha=1.0, label='hp', dis_th=0.5) # Use only softmax
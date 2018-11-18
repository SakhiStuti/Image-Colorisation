import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from model import ColorizationNetwork
from skimage.color import lab2rgb
from skimage import io
import os

gamut = np.load('./prior_prob/pts_in_gamut.npy')


class training:

    def __init__(self, args):
        self.model = ColorizationNetwork()
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.num_iterations = args.num_iterations
        self.gpu = args.gpu
        self.pretrained = args.pretrained
        self.epoch = args.epoch
        self.save_directory = args.save_directory
        self.resume = args.resume
        self.lr = args.lr
        self.lr_update_iter = args.lr_update_iter
        self.loss_arr = []
        #define criterion
        self.criterion = nn.CrossEntropyLoss(reduction =False).cuda()
        #optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-3)

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def save_imgs(tensor, filename):
        for index, im in enumerate(tensor):
            # print(im.shape)
            # im =np.clip(im.numpy().transpose(1,2,0), -1, 1) 
            img_rgb_out = (255*np.clip(lab2rgb(im),0,1)).astype('uint8')
            io.imsave(filename +'rgb'+ str(index) + '.png', img_rgb_out )

    def restore(self,resume_iter):
        path = os.path.join(self.save_directory, './net_%d.pth'%(resume_iter))
        checkpoint_dict = torch.load(path)
        self.model.load_state_dict(checkpoint_dict['state_dict'])
        self.lr = checkpoint_dict['lr']
        return checkpoint_dict['iteration']


    def train(self, train_loader, val_loader):

        start_iter = 0
        if self.resume:
            # TODO
            start_iter = self.restore(self.resume)
        #transfer to GPU
        
        model = nn.DataParallel(self.model).cuda()

        #Set to training mode
        model.train()

        

        print('..........................Starting training.................')

        #start training
        for i, data in enumerate(train_loader, start_iter):
            img, _ = data

            img = Variable(img).cuda()

            weights, Z_gt, Z_pred = model(img, self.pretrained)

            loss = ((self.criterion(Z_gt, Z_pred).sum(axis = 1))*weights.squeeze(axis = 1)).sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #logging
            if (i) % 500 == 0:
                print('At iteration %d, loss is %.4f'%(i,loss.data()))
                self.loss_arr.append(loss.data())

            #Update learning rate if required
            if (i) % self.lr_update_iter == 0:
                self.record_iters = i
                if self.lr > 1e-8:
                    self.lr *= 0.316
                self.update_lr(self.lr)

            #validate/Test
            if i%5000 ==0:
                self.test(val_loader, i)


            #checkpoint
            if (i)%2000 == 0:
                state_dict = model.state_dict()
                checkpoint = {'iteration': i,
                              'state_dict': state_dict,
                              'lr' : self.lr}
                save_path = os.path.join(self.save_directory, './net_%d.pth'%(i+1))
                torch.save(checkpoint, save_path)

        print('...............Training Completed...........')
        
        
    def test(self, test_loader, curr_iter, inference_iter=0):
        
        # Load the trained generator.
        self.optimizer.zero_grad()
        data_iter = iter(test_loader)

        if inference_iter:
            self.restore(inference_iter)

        if inference_iter:
            print('Start inferencing...')
        else:
            print('Start Validating...')
            
        
        self.model.eval()  # Set g_model to training mode

        img_dir =  os.path.join(self.save_directory, 'Test','%d/' % (self.curr_iter))
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        len_record = len(test_loader)
        softmax_op = torch.nn.Softmax()
        test_loss = 0.0

        for global_iteration in range(len_record):
            
            
            print('completed %d of %d' % (global_iteration, len_record))
            
            # Iterate over data.
            img , _ = next(data_iter)

            # wrap them in Variable
            if self.cuda:
                img = Variable(img.cuda(), volatile=True)
            else:
                img = Variable(img, volatile=True) 

            weights, Z_gt, Z_pred, Z_pred_upsample  = self.model(img)
            loss = ((self.criterion(Z_gt, Z_pred).sum(axis = 1))*weights.squeeze(axis = 1)).sum()
            test_loss += loss.data[0]

            img_L = img[:,:1,:,:] #[batch, 1, 224, 224]

            # post-process
            Z_pred_upsample *= 2.606
            Z_pred_upsample = softmax_op(Z_pred_upsample).cpu().data.numpy()

            fac_a = gamut[:,0][np.newaxis,:,np.newaxis,np.newaxis]
            fac_b = gamut[:,1][np.newaxis,:,np.newaxis,np.newaxis]

            img_L = img_L.cpu().data.numpy().transpose(0,2,3,1) #[batch, 224, 224, 1]
            frs_pred_ab = np.concatenate((np.sum(Z_pred_upsample * fac_a, axis=1, keepdims=True), np.sum(Z_pred_upsample * fac_b, axis=1, keepdims=True)), axis=1).transpose(0,2,3,1)
            #[batch, 224, 224, 2]
            
            frs_predic_imgs = np.concatenate((img_L, frs_pred_ab ), axis = 3) #[batch, 224, 224, 3]
            print('Saving image %s%d_frspredic_' %  (img_dir, global_iteration))
            self.save_imgs(frs_predic_imgs, '%s%d_frspredic_' %  (img_dir, global_iteration))
            img = img.cpu().data.numpy().transpose(0,2,3,1).astype('float64')
            self.save_imgs(img,'%s%d_img_' %  (img_dir ,global_iteration))





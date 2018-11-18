import torchvision
import torch
from torch.autograd import Variable
import torch.optim as optim

from CrossEntropyLoss import CE_loss

class training:

    def __init__(self, args):
        self.model = args.model
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
        #optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-3)

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def restore(self,resume_iter):
        path = os.path.join(self.save_directory, './net_%d.pth'%(resume_iter))
        checkpoint_dict = torch.load(path)
        self.model.load_state_dict(checkpoint_dict['state_dict'])
        self.lr = checkpoint_dict['lr']
        return checkpoint_dict['iteration']


    def train(self, train_loader, val_loader):

        train_iter = iter(train_loader)

        start_iter = 0
        if self.resume:
            # TODO
            start_iter = self.restore(self.resume)
        #transfer to GPU
        model = self.model.cuda()

        #Set to training mode
        model.train()

        #define loss
        criteria = CE_loss()

        print('..........................Starting training.................')

        #start training
        for i, data in train_loader:
            img, _ = data

            img = Variable(img).cuda()

            output, enc = model(img)

            loss = criteria(output, enc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #logging
            if (i+1) % 500 == 0:
                print('At iteration %d, loss is %.4f'%(i,loss.data())
                self.loss_arr.append(loss.data())

            #Update learning rate if required
            if (i+1) % self.lr_update_iter == 0:
                self.record_iters = i
                if self.lr > 1e-8:
                    self.lr *= 0.316
                self.update_lr(self.lr)

            #validate/Test



            #checkpoint
            if (i+1)%2000 == 0:
                state_dict = model.state_dict()
                checkpoint = {iteration: i,
                              state_dict: state_dict,
                              lr : self.lr}
                save_path = os.path.join(self.save_directory, './net_%d.pth'%(i+1))
                torch.save(checkpoint, save_path)

        print('...............Training Completed...........')





import torchvision

class training:

    def __init__(self, args):
        self.args = args

    def train(self, train_loader, val_loader):

        train_iter = iter(train_loader)
        data, label  = train_iter.next()

        val_iter = iter(val_loader)
        data2, label  = val_iter.next()

        print(data.shape, '    ', data2.shape)
        torchvision.utils.save_image(data2,'test1.png', nrow=2)

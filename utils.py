import torch.cuda
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os

def weights_init(m):
    classname=m.__class__.__name__
    # print(classname)
    if classname.find('Conv')!=-1:
        m.weight.data.normal_(0.0,0.1)
    elif classname.find('Linear')!=-1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm')!=-1:
        m.weight.data.normal_(1.0,0.1)
        m.bias.data.fill_(0)

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x=x.cuda()
    return Variable(x,requires_grad=requires_grad)

# flag=0
def inv_lr_scheduler(param_lr,optimizer,iter_num,
                     gamma=0.0001,power=0.75,init_lr=0.001):
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    # if flag==0:
    #     if iter_num>300:
    #         lr=lr*0.1
    #         flag=1
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr']=lr*param_lr[i]
        i+=1
    return optimizer


def plot_acc_loss(filepath,val_interval,net,loss, acc):
    acc_,=plt.plot([i*val_interval for i in range(1,1+len(acc))],acc,label="accuracy")
    loss_,=plt.plot([i*val_interval for i in range(1,1+len(loss))], loss, label="loss")
    plt.title('validation accuracy&loss')
    plt.xlabel('epoches')
    plt.ylabel('accuracy&loss')

    # Create a legend for the first line.
    first_legend = plt.legend(handles=[acc_], loc=1)

    # Add the legend manually to the current Axes.
    ax = plt.gca().add_artist(first_legend)

    # Create another legend for the second line.
    plt.legend(handles=[loss_], loc=4)

    plt.draw()
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    path=os.path.join(filepath,'epoch%d_acc_loss.jpg'%(val_interval*len(loss)))
    plt.savefig(path)
    plt.show()

def plot_acc(filepath,val_interval, acc, status):
    acc_,=plt.plot([i*val_interval for i in range(1,1+len(acc))],acc,label="accuracy")
    plt.title(status+' accuracy')
    plt.xlabel('epoches')
    plt.ylabel(status+' accuracy')
    plt.draw()
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    path=os.path.join(filepath,status+'epoch%d_acc.jpg'%(val_interval*len(acc)))
    plt.savefig(path)
    plt.show()


def plot_loss(filepath,val_interval,loss,status):
    loss_,=plt.plot([i*val_interval for i in range(1,1+len(loss))], loss, label="loss")
    plt.title(status+' loss')
    plt.xlabel('epoches')
    plt.ylabel(status+' loss')
    plt.draw()
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    path=os.path.join(filepath,status+'epoch%d_loss.jpg'%(val_interval*len(loss)))
    plt.savefig(path)
    plt.show()


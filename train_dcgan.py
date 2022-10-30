from random import shuffle
import torch, torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pylab
from utils import *
import os

FIGS_DIR = './figs/'
SAVE_DIR = './save/'
DATA_DIR = '/home/wangqs/Data/'

batch_size = 16
num_epochs = 1000
num_G_step = 1
num_D_step = 1
use_cuda = True
img_size = 64
latent_dim = 100

g_lr = 1e-3
d_lr = 1e-3
weight_decay = 1e-4
gamma = 0.9

if not os.path.exists(FIGS_DIR):
    os.makedirs(FIGS_DIR)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5),   
                                         std=(0.5, 0.5, 0.5))])


train_dataset = POKEMON(root_dir=DATA_DIR, img_size=img_size, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def recover(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


use_cuda = use_cuda and torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

G = DCGenerator(nz=latent_dim).to(device)
D = DCDiscriminator().to(device)

criterion = nn.BCELoss()
d_optimizer = torch.optim.SGD(D.parameters(), lr=d_lr, weight_decay=weight_decay)
d_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=d_optimizer, step_size=10, gamma=gamma)
g_optimizer = torch.optim.SGD(G.parameters(), lr=g_lr, weight_decay=weight_decay)
g_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=g_optimizer, step_size=10, gamma=gamma)

d_losses = np.zeros(num_epochs)
g_losses = np.zeros(num_epochs)
real_scores = np.zeros(num_epochs)
fake_scores = np.zeros(num_epochs)

num_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, images in enumerate(train_loader):

        if images.size(0) < batch_size:
            continue
        else:
            temp = images
        images = images.to(device)
        # images = Variable(images)
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)
        # real_labels = Variable(real_labels)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        # fake_labels = Variable(fake_labels)

        for k in range(num_D_step):
            outputs = D(images)
            # print(outputs.size(), real_labels.size())
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs

            z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            # z = Variable(z)
            fake_images = G(z)
            outputs = D(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
        
        for j in range(num_G_step):
            z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            #z = Variable(z)
            fake_images = G(z)
            outputs = D(fake_images)

            g_loss = criterion(outputs, real_labels)

            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        d_losses[epoch] = d_losses[epoch]*(i/(i+1.)) + d_loss.data.item()*(1./(i+1.))
        g_losses[epoch] = g_losses[epoch]*(i/(i+1.)) + g_loss.data.item()*(1./(i+1.))
        real_scores[epoch] = real_scores[epoch]*(i/(i+1.)) + real_score.mean().data.item()*(1./(i+1.))
        fake_scores[epoch] = fake_scores[epoch]*(i/(i+1.)) + fake_score.mean().data.item()*(1./(i+1.))

        if (i+1) == num_steps - 1:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                    .format(epoch, num_epochs, i+1, num_steps, d_loss.data.item(), g_loss.data.item(), 
                            real_score.mean().data.item(), fake_score.mean().data.item()))

    d_scheduler.step()
    g_scheduler.step()

    if epoch == 0:
        save_image(recover(temp.data), os.path.join(FIGS_DIR, 'real_images.png'))

    if (epoch + 1) % 50 == 0:
        fake_images = fake_images.view(batch_size, 3, img_size, img_size)
        save_image(recover(fake_images.data), os.path.join(FIGS_DIR, 'fake_images-{}.png'.format(epoch+1)))

    plt.figure()
    pylab.xlim(0, num_epochs + 1)
    plt.plot(range(1, num_epochs + 1), d_losses, label='d loss')
    plt.plot(range(1, num_epochs + 1), g_losses, label='g loss')    
    plt.legend()
    plt.savefig(os.path.join(FIGS_DIR, 'loss.png'))
    plt.close()

    plt.figure()
    pylab.xlim(0, num_epochs + 1)
    pylab.ylim(0, 1)
    plt.plot(range(1, num_epochs + 1), fake_scores, label='fake score')
    plt.plot(range(1, num_epochs + 1), real_scores, label='real score')    
    plt.legend()
    plt.savefig(os.path.join(FIGS_DIR, 'accuracy.png'))
    plt.close()

    if (epoch+1) % 50 == 0:
        torch.save(G.state_dict(), os.path.join(SAVE_DIR, 'G--{}.ckpt'.format(epoch+1)))
        torch.save(D.state_dict(), os.path.join(SAVE_DIR, 'D--{}.ckpt'.format(epoch+1)))


torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')
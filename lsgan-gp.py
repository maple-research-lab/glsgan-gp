import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import torch.autograd as autograd
import os
import matplotlib.pyplot as plt

# Run this command to execute the script.
# python lsgan-gp.py --dataset folder --dataroot celebA --cuda --niter 25

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--result', default='samples', help='output folder')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for Critic, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for Generator, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--lamb', type=float, default=0.0002, help='the scale of the distance metric used for adaptive margins. This is actually tau in the original paper. L2: 0.05/L1: 0.001, temporary best 0.008 before applying scaling')
parser.add_argument('--gamma', type=float, default=10, help='the scale of gradient penalty')
parser.add_argument('--optim_method', type=int, default=1, help='Whether to use adam (default is adam)')
opt = parser.parse_args()

print("-------- folder --------")
ouputPath = os.getcwd() + "/" + opt.result
print(os.path.exists(ouputPath))
if not os.path.exists(ouputPath):
    print("Creating folder.")
    os.mkdir(ouputPath)

print("-------- parameters --------")
print(opt)

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )

assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        # Add no bias
        
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)

# Net G
print("-------- netG --------")
netG = nn.Sequential(
        #
        nn.ConvTranspose2d(nz, ngf*8, 4, 4),
        nn.BatchNorm2d(ngf*8),
        nn.ReLU(True),

        #
        nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1),
        nn.BatchNorm2d(ngf*4),
        nn.ReLU(True),

        #
        nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1),
        nn.BatchNorm2d(ngf*2),
        nn.ReLU(True),

        #
        nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),

        #
        nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
        nn.Tanh()
    )

netG.apply(weights_init)

print(netG)

# Net D
print("-------- netD --------")
netD = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1),
            #nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),
            #nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1),
            #nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*8, 1, 4)
        )

netD.apply(weights_init)

print(netD)

# Graident penalty proposed originally in Qi's paper.
def calc_gradient_penalty(netD, real_data, fake_data):
    alpha1 = real_data.clone().fill_(random.uniform(0, 1))
    alpha1 = alpha1.cuda() if opt.cuda else alpha1

    alpha2 = real_data.clone().fill_(random.uniform(0, 1))
    alpha2 = alpha2.cuda() if opt.cuda else alpha2

    x1 = autograd.Variable(alpha1 * real_data + ((1 - alpha1) * fake_data))
    x2 = autograd.Variable(alpha2 * real_data + ((1 - alpha2) * fake_data))
    y1 = netD(x1)
    y2 = netD(x2)

    dist = nn.PairwiseDistance(2)
    x1 = x1.view(x1.size(0), x1.size(1) * x1.size(2) * x1.size(3)) 
    x2 = x2.view(x2.size(0), x2.size(1) * x2.size(2) * x2.size(3))

    gradient_penalty = ((y1 - y2).abs()/dist(x1, x2)).mean() * opt.gamma
    
    '''
    # If you are using the master version of Pytorch, please replace the code above.
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if opt.cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    # The following function is only supported in the master branch of pytorch.
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if opt.cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.gamma
    '''
    
    return gradient_penalty

# Gradient penalty proposed originally in wgan-gp.
def calc_gradient_penalty_wgan_gp(netD, real_data, fake_data):
    alpha1 = real_data.clone().fill_(random.uniform(0, 1))
    alpha1 = alpha1.cuda() if opt.cuda else alpha1

    alpha2 = real_data.clone().fill_(random.uniform(0, 1))
    alpha2 = alpha2.cuda() if opt.cuda else alpha2

    x1 = autograd.Variable(alpha1 * real_data + ((1 - alpha1) * fake_data))
    x2 = autograd.Variable(alpha2 * real_data + ((1 - alpha2) * fake_data))
    y1 = netD(x1)
    y2 = netD(x2)

    dist = nn.PairwiseDistance(2)
    x1 = x1.view(x1.size(0), x1.size(1) * x1.size(2) * x1.size(3)) 
    x2 = x2.view(x2.size(0), x2.size(1) * x2.size(2) * x2.size(3))

    gradient_penalty = (((y1 - y2).abs()/dist(x1, x2) -1) **2).mean() * opt.gamma
    return gradient_penalty

# --------- optimizer --------
if opt.optim_method == 1:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
elif opt.optim_method == 2:
    optimizerD = optim.RMSprop(netD.parameters(), lr = 0.00005)
    optimizerG = optim.RMSprop(netG.parameters(), lr = 0.00005)
else:
    print("Wrong optimizer!")

# -------- Init tensor --------
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
fake = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)

l1dist = nn.PairwiseDistance(1)
l2dist = nn.PairwiseDistance(2)
hingeLoss = nn.HingeEmbeddingLoss()

# -------- Run on GPU --------
if opt.cuda:
    netD.cuda()
    netG.cuda()
    
    input = input.cuda()
    fake = fake.cuda()
    noise = noise.cuda()

    l1dist = l1dist.cuda()
    hingeLoss = hingeLoss.cuda()

# -------- Training --------
for epoch in range(opt.niter):
    data_iter = iter(dataloader)

    i = 0
    while i < len(dataloader):
        i = i + 1

        data = data_iter.next()
        
        # Update D network
        for p in netD.parameters():
            p.requires_grad = True 

        netD.zero_grad()

        # train with real
        input , _ = data
        dataSize = input.size(0)
        noise = torch.FloatTensor(dataSize, nz, 1, 1)
        
        if opt.cuda:
            input = input.cuda()
            noise = noise.cuda()

        inputv = autograd.Variable(input, requires_grad=True)

        # Error R of real
        outputR = netD(inputv)

        # Create noise and freeze net G.
        noise.normal_(0, 1)
        noisev = autograd.Variable(noise)

        # Train with fake
        # -----------------------------------------------------------------#     
        
        # Loss for real and fake
        fake = autograd.Variable(netG(noisev).data)
        outputF = netD(fake)

        # L1 distance
        pdist = l1dist(
            inputv.view(dataSize, opt.nc * opt.imageSize * opt.imageSize), 
            fake.view(dataSize, opt.nc * opt.imageSize * opt.imageSize))
        
        pdist = pdist.mul(opt.lamb)

        # Loss function for D.
        cost = outputR - outputF + pdist
        
        # Calculate hinge loss.
        df_error_hinge = cost.clone()
        df_error_hinge.data = df_error_hinge.data.fill_(1.0)
        df_error_hinge.data[cost.data.le(0)] = 0.0
        df_error_hinge = df_error_hinge/dataSize
        
        if opt.cuda:
            df_error_hinge = df_error_hinge.cuda()

        # Train with fake
        outputF.backward(df_error_hinge.data * -1) 

        # Train with real.
        outputR = netD(inputv)
        outputR.backward(df_error_hinge.data)

        # Gradient of D
        gradD = inputv.grad.mean().abs()

        # Train with gradient penalty
        # -----------------------------------------------------------------#
        gp = calc_gradient_penalty(netD, inputv.data, fake.data)
        gp.backward()

        # Error of D
        errD = cost.mean()

        # Automatically accumulate gradients.
        optimizerD.step()

        # Update G network
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation

        netG.zero_grad()

        noise.normal_(0, 1)
        noisev = autograd.Variable(noise, requires_grad=True)

        fake = netG(noisev)
        outputF = netD(fake)

        df_error_hinge= outputF.clone()
        df_error_hinge.data.fill_(1.0)
        df_error_hinge = df_error_hinge/(dataSize)
        
        if opt.cuda:
            df_error_hinge = df_error_hinge.cuda()

        outputF.backward(df_error_hinge.data)

        # Error of G
        errG = outputF.mean()

        # Gradient of G
        gradG = noisev.grad.mean().abs()

        # Automatically accumulate gradients.
        optimizerG.step()

        print('Epoch {}, [{}/{}], ErrG {:.4f}, ErrD {:.4f}, OutputR {:.4f}, OutputF {:.4f}, distD {:.4f}, gradD {:.8f}, gradG {:.8f}, gp {:.4f}'.format(
                epoch, 
                i, 
                len(dataloader), 
                errG.data[0], 
                errD.data[0], 
                torch.mean(outputR).data[0], 
                torch.mean(outputF).data[0], 
                torch.mean(pdist).data[0], 
                gradD.data[0], 
                gradG.data[0],
                gp.data.mean()))

        if i % 100 == 0:
            input = input.mul(0.5).add(0.5)
            vutils.save_image(input, ouputPath + '/real_samples_{:02d}_{:08d}.jpg'.format(epoch, i))

            fake.data = fake.data.mul(0.5).add(0.5)
            vutils.save_image(fake.data, ouputPath + '/fake_samples_{:02d}_{:08d}.jpg'.format(epoch, i))
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
parser.add_argument('--lrD', type=float, default=2e-4, help='learning rate for Critic, default=0.0002')
parser.add_argument('--lrG', type=float, default=2e-4, help='learning rate for Generator, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--gpu', type=int, default=0, help='run on selected gpu, default is 0')
parser.add_argument('--lamb', type=float, default=2e-4, help='the scale of the distance metric used for adaptive margins. This is actually tau in the original paper. L2: 0.05/L1: 0.001, temporary best 0.008 before applying scaling')
parser.add_argument('--gamma', type=float, default=10, help='the scale of gradient penalty')
parser.add_argument('--slope', type=float, default=0.0, help='slope for function c proposed in generalized lsgan, when slope is 0, gls-gan is lsgan, when slope is 1 gls-gan is wgan.')
parser.add_argument('--optim_method', type=int, default=1, help='Whether to use adam (default is adam)')
parser.add_argument('--manualSeed', type=int, help='manual seed')
opt = parser.parse_args()

print("-------- folder --------")
ouputPath = os.getcwd() + "/" + opt.result
if not os.path.exists(ouputPath):
    print("Creating folder.")
    os.mkdir(ouputPath)

print("-------- parameters --------")
print(opt)

opt.manualSeed = random.randint(1, 10000)
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

# Weights initialization.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        
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
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*8, 1, 4)
        )

netD.apply(weights_init)
print(netD)

# Graident penalty proposed originally in Qi's paper.
def get_direct_gradient_penalty(netD, x, gamma, cuda):
    if cuda:
        x = x.cuda()

    x = autograd.Variable(x, requires_grad=True)
    output = netD(x)
    gradOutput = torch.ones(output.size()).cuda() if cuda else torch.ones(output.size())
    
    gradient = torch.autograd.grad(outputs=output, inputs=x, grad_outputs=gradOutput, create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradientPenalty = (gradient.norm(2, dim=1)).mean() * gamma
    
    return gradientPenalty

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
l1dist = nn.PairwiseDistance(1)
l2dist = nn.PairwiseDistance(2)
LeakyReLU = nn.LeakyReLU(opt.slope)

with torch.cuda.device(opt.gpu):
    print("--------GPU Config--------")
    print("Current GPU: " + str(torch.cuda.current_device()))
    print("Total GPU: " + str(torch.cuda.device_count()))

    # -------- Run on GPU --------
    if opt.cuda:
        netD.cuda()
        netG.cuda()

        l1dist = l1dist.cuda()
        l2dist = l2dist.cuda()
        LeakyReLU = LeakyReLU.cuda()

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

            # Get batch data and initialize parameters.
            x , _ = data
            dataSize = x.size(0)
            z = torch.FloatTensor(dataSize, nz, 1, 1).normal_(0, 1)
            
            if opt.cuda:
                x = x.cuda()
                z = z.cuda()

            x = autograd.Variable(x, requires_grad=True)
            z = autograd.Variable(z)

            fake = autograd.Variable(netG(z).data)

            # Loss R for real
            outputR = netD(x)
            
            # Loss F for fake.
            outputF = netD(fake)

            # L1 distance between real and fake.
            pdist = l1dist(x.view(dataSize, -1), fake.view(dataSize, -1)).mul(opt.lamb)

            # Loss for D.
            errD = LeakyReLU(outputR - outputF + pdist).mean()
            errD.backward()
            
            # Penalize direct gradient of x.
            gp = get_direct_gradient_penalty(netD, x.data, opt.gamma, opt.cuda)
            gp.backward()

            # Gradient of D.
            gradD = x.grad

            # Automatically accumulate gradients.
            optimizerD.step()

            # Update G network, freeze D.
            for p in netD.parameters():
                p.requires_grad = False 

            netG.zero_grad()

            # Create noise with normal distribution and project into data space.
            z = torch.FloatTensor(dataSize, nz, 1, 1).normal_(0, 1)
            
            if opt.cuda:
                z = z.cuda()

            z = autograd.Variable(z, requires_grad=True)
            fake = netG(z)

            # Loss F for fake.
            outputF = netD(fake)

            # Error of G.
            errG = outputF.mean()
            errG.backward()

            # Gradient of G.
            gradG = z.grad

            # Automatically accumulate gradients.
            optimizerG.step()

            # Showing debugging information.
            print('Epoch {}, [{}/{}], ErrD {:.4f}, ErrG {:.4f}, OutputR {:.4f}, OutputF {:.4f}, DistD {:.4f}, GradD {:.8f}, GradG {:.8f}, GP {:.4f}'.format(
                    epoch+1, 
                    i, 
                    len(dataloader), 
                    errD.data.mean(), 
                    errG.data.mean(),
                    outputR.data.mean(), 
                    outputF.data.mean(), 
                    pdist.data.mean(), 
                    gradD.data.abs().mean(), 
                    gradG.data.abs().mean(),
                    gp.data.mean()))

            if i % 100 == 0:
                x.data = x.data.mul(0.5).add(0.5)
                vutils.save_image(x.data, ouputPath + '/real_samples_{:02d}_{:08d}.jpg'.format(epoch, i))

                fake.data = fake.data.mul(0.5).add(0.5)
                vutils.save_image(fake.data, ouputPath + '/fake_samples_{:02d}_{:08d}.jpg'.format(epoch, i))
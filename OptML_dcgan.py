import numpy as np
import torch
import torchvision
from torch import nn, optim
from torchvision.transforms import transforms

"""
Code adapted from pytorch examples to custom training loop and optimizers
https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
"""

def prepare_dataset(batch_size=128, num_workers=4):
    dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return dataloader


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1, 0.02)
        nn.init.constant_(m.bias.data, 0.)


class Generator(nn.Module):
    def __init__(self, noise_dim=100, feature_mult=64, channels=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(noise_dim, feature_mult * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_mult * 8),
            nn.ReLU(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(feature_mult * 8, feature_mult * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_mult * 4),
            nn.ReLU(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(feature_mult * 4, feature_mult * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_mult * 2),
            nn.ReLU(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(feature_mult * 2, feature_mult, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_mult),
            nn.ReLU(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(feature_mult, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, channels=3, features_mult=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(channels, features_mult, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(features_mult, features_mult * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_mult * 2),
            nn.LeakyReLU(0.2),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(features_mult * 2, features_mult * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_mult * 4),
            nn.LeakyReLU(0.2),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(features_mult * 4, features_mult * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_mult * 8),
            nn.LeakyReLU(0.2),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(features_mult * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


def training_step(
        gen,
        dis,
        criterion,
        g_optim,
        d_optim,
        batch,
        noise_dim,
        device,
):

    x_train, _ = batch
    gen.train()
    dis.train()
    bs = x_train.shape[0]
    x_train.to(device)
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # Train with all-real batch
    dis.zero_grad()
    # Format batch
    labels = torch.ones((bs, ), device=device)
    # Forward pass real batch through D
    output = dis(x_train).view(-1)
    # Calculate loss on all-real batch
    dis_real_loss = criterion(output, labels)
    # Calculate gradients for D in backward pass
    dis_real_loss.backward()

    # Train with all-fake batch
    # Generate batch of latent vectors
    noise = torch.randn(bs, noise_dim, 1, 1, device=device)
    # Generate fake image batch with G
    fake = gen(noise)
    labels.fill_(0)
    # Classify all fake batch with D
    output = dis(fake.detach()).view(-1)
    # Calculate D's loss on the all-fake batch
    dis_fake_loss = criterion(output, labels)
    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
    dis_fake_loss.backward()
    # Compute error of D as sum over the fake and the real batches
    dis_loss = dis_real_loss + dis_fake_loss
    # Update D
    d_optim.step()


    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    gen.zero_grad()
    labels.fill_(1)  # fake labels are real for generator cost
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = dis(fake).view(-1)
    # Calculate G's loss based on this output
    gen_loss = criterion(output, labels)
    # Calculate gradients for G
    gen_loss.backward()
    # Update G
    g_optim.step()

    return {
        "dis_loss": dis_loss.detach().cpu().item(),
        "gen_loss": gen_loss.detach().cpu().item(),
    }


# Training loop
noise_dim = 100
lr = 0.0002
beta1 = 0.5
epoch = 5
log_every = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
dataloader = prepare_dataset()
criterion = nn.BCELoss()
T = 3

generator = Generator()
discriminator = Discriminator()

generator.apply(weight_init)
discriminator.apply(weight_init)

optim_d_1 = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optim_g_1 = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

optim_d_2 = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optim_g_2 = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

dis_loss = []
gen_loss = []

for e in range(epoch):
    for idx, batch in enumerate(dataloader):
        if e < T:
            optim_g = optim_g_1
            optim_d = optim_d_1
        else:
            optim_g = optim_g_2
            optim_d = optim_d_2

        logs = training_step(
            generator,
            discriminator,
            criterion,
            optim_g,
            optim_d,
            batch,
            noise_dim,
            device
        )

        dis_loss.append(logs['dis_loss'])
        gen_loss.append(logs['gen_loss'])

        if idx % log_every == 0:
            print(
                f"[{e}/{epoch}]"
                f"[{idx}/{len(dataloader)}]"
                f"\tdis-los: {logs['dis_loss'] :.4f}"
                f"\tgen-loss: {logs['gen_loss'] :.4f}")


np.save('123123.npy', dis_loss)
np.save('123123123.npy', gen_loss)


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.generator import Generator
# from models.discriminator import Discriminator
from models.discriminator import Critic
from utils.save_images import save_generated_images


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("outputs", exist_ok=True)

    latent_dim = 100
    batch_size = 128
    epochs = 30
    lr=1e-4

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    generator = Generator(latent_dim=latent_dim).to(device)
    # discriminator = Discriminator().to(device)
    critic = Critic().to(device)
    generator.apply(weights_init)
    # discriminator.apply(weights_init)
    critic.apply(weights_init)
    # criterion = nn.BCELoss()

    # optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    # optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    # optimizer_g = optim.RMSprop(generator.parameters(), lr=lr)
    # optimizer_c = optim.RMSprop(critic.parameters(), lr=lr)
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
    optimizer_c = optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))
    fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)
    # for epoch in range(epochs):
    #     for batch_idx, (real_images, _) in enumerate(dataloader):
    #         real_images = real_images.to(device)
    #         batch_size_curr = real_images.size(0)
    #
    #         real_labels = torch.ones(batch_size_curr, device=device)
    #         fake_labels = torch.zeros(batch_size_curr, device=device)
    #
    #         # train discriminator
    #         noise = torch.randn(batch_size_curr, latent_dim, 1, 1, device=device)
    #         fake_images = generator(noise)
    #
    #         d_real = discriminator(real_images)
    #         loss_d_real = criterion(d_real, real_labels)
    #
    #         d_fake = discriminator(fake_images.detach())
    #         loss_d_fake = criterion(d_fake, fake_labels)
    #
    #         loss_d = loss_d_real + loss_d_fake
    #
    #         optimizer_d.zero_grad()
    #         loss_d.backward()
    #         optimizer_d.step()
    #
    #         # train generator
    #         output = discriminator(fake_images)
    #         loss_g = criterion(output, real_labels)
    #
    #         optimizer_g.zero_grad()
    #         loss_g.backward()
    #         optimizer_g.step()
    #
    #         if batch_idx % 100 == 0:
    #             print(
    #                 f"Epoch [{epoch+1}/{epochs}] "
    #                 f"Batch [{batch_idx}/{len(dataloader)}] "
    #                 f"Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}"
    #             )
    #
    # with torch.no_grad():
    #     fake_samples = generator(fixed_noise)
    #     save_generated_images(fake_samples, epoch + 1)

  # WGAN

    n_critic = 5  # 每次训练5次critic

    # for epoch in range(epochs):
    #     for batch_idx, (real_images, _) in enumerate(dataloader):
    #
    #         real_images = real_images.to(device)
    #         batch_size_curr = real_images.size(0)
    #
    #         # ---------------------
    #         # Train Critic
    #         # ---------------------
    #         for _ in range(n_critic):
    #             noise = torch.randn(batch_size_curr, latent_dim, 1, 1, device=device)
    #             fake_images = generator(noise)
    #
    #             critic_real = critic(real_images)
    #             critic_fake = critic(fake_images.detach())
    #
    #             loss_c = -(torch.mean(critic_real) - torch.mean(critic_fake))
    #
    #             optimizer_c.zero_grad()
    #             loss_c.backward()
    #             optimizer_c.step()
    #
    #             # weight clipping
    #             for p in critic.parameters():
    #                 p.data.clamp_(-0.01, 0.01)
    #
    #         # ---------------------
    #         # Train Generator
    #         # ---------------------
    #         noise = torch.randn(batch_size_curr, latent_dim, 1, 1, device=device)
    #         fake_images = generator(noise)
    #
    #         loss_g = -torch.mean(critic(fake_images))
    #
    #         optimizer_g.zero_grad()
    #         loss_g.backward()
    #         optimizer_g.step()
    #
    #         if batch_idx % 100 == 0:
    #             print(
    #                 f"Epoch [{epoch + 1}/{epochs}] "
    #                 f"Batch [{batch_idx}/{len(dataloader)}] "
    #                 f"Loss C: {loss_c.item():.4f}, Loss G: {loss_g.item():.4f}"
    #             )

    # WGAN-GP
    n_critic = 5
    lambda_gp = 10

    for epoch in range(epochs):
        for batch_idx, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size_curr = real_images.size(0)

            # ---------------------
            # Train Critic
            # ---------------------
            for _ in range(n_critic):
                noise = torch.randn(batch_size_curr, latent_dim, 1, 1, device=device)
                fake_images = generator(noise)

                critic_real = critic(real_images)
                critic_fake = critic(fake_images.detach())

                gp = gradient_penalty(critic, real_images, fake_images.detach(), device)

                loss_c = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp

                optimizer_c.zero_grad()
                loss_c.backward()
                optimizer_c.step()

            # ---------------------
            # Train Generator
            # ---------------------
            noise = torch.randn(batch_size_curr, latent_dim, 1, 1, device=device)
            fake_images = generator(noise)

            loss_g = -torch.mean(critic(fake_images))

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss C: {loss_c.item():.4f}, "
                    f"Loss G: {loss_g.item():.4f}, "
                    f"GP: {gp.item():.4f}"
                )
            with torch.no_grad():
                fake_samples = generator(fixed_noise)
                save_generated_images(fake_samples, epoch + 1)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(generator.state_dict(), "checkpoints/generator_critic.pth")
    torch.save(critic.state_dict(), "checkpoints/critic.pth")
    print("Training finished.")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)

    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated)

    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradient = gradient.view(batch_size, -1)
    gradient_norm = gradient.norm(2, dim=1)

    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp


if __name__ == "__main__":
    main()

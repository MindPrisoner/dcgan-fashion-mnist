import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.generator import Generator
from models.discriminator import Discriminator
from utils.save_images import save_generated_images


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("outputs", exist_ok=True)

    latent_dim = 100
    batch_size = 128
    epochs = 20
    lr = 0.0002

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
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()

    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)

    for epoch in range(epochs):
        for batch_idx, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size_curr = real_images.size(0)

            real_labels = torch.ones(batch_size_curr, device=device)
            fake_labels = torch.zeros(batch_size_curr, device=device)

            # train discriminator
            noise = torch.randn(batch_size_curr, latent_dim, 1, 1, device=device)
            fake_images = generator(noise)

            d_real = discriminator(real_images)
            loss_d_real = criterion(d_real, real_labels)

            d_fake = discriminator(fake_images.detach())
            loss_d_fake = criterion(d_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            # train generator
            output = discriminator(fake_images)
            loss_g = criterion(output, real_labels)

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}"
                )

        with torch.no_grad():
            fake_samples = generator(fixed_noise)
            save_generated_images(fake_samples, epoch + 1)

    print("Training finished.")


if __name__ == "__main__":
    main()

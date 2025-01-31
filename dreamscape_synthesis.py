import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model
from torchvision import transforms
from PIL import Image
import os

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
LATENT_DIM = 256
IMAGE_SIZE = 256
TEXT_EMBEDDING_DIM = 768

# Dataset
class DreamScapeDataset(Dataset):
    def __init__(self, text_file, image_dir, transform=None):
        self.text_file = text_file
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        with open(text_file, 'r') as f:
            self.descriptions = f.readlines()

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        description = self.descriptions[idx].strip()
        image_path = os.path.join(self.image_dir, f"{idx}.png")
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        tokens = self.tokenizer(description, return_tensors='pt', padding='max_length', max_length=50, truncation=True)
        return tokens['input_ids'].squeeze(0), image

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.text_encoder = GPT2Model.from_pretrained('gpt2')
        self.fc = nn.Linear(TEXT_EMBEDDING_DIM, LATENT_DIM)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, text_input):
        text_embedding = self.text_encoder(text_input).last_hidden_state.mean(dim=1)
        latent_vector = self.fc(text_embedding)
        latent_vector = latent_vector.view(-1, LATENT_DIM, 1, 1)
        image = self.deconv(latent_vector)
        return image

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, image):
        validity = self.conv(image)
        return validity.view(-1, 1)

# Training
def train(dataloader, generator, discriminator, optimizer_G, optimizer_D, criterion, device):
    generator.train()
    discriminator.train()

    for epoch in range(EPOCHS):
        for i, (text_input, real_images) in enumerate(dataloader):
            real_images = real_images.to(device)
            text_input = text_input.to(device)

            # Train Discriminator
            optimizer_D.zero_grad()
            real_labels = torch.ones((real_images.size(0), 1), device=device)
            fake_labels = torch.zeros((real_images.size(0), 1), device=device)

            real_loss = criterion(discriminator(real_images), real_labels)
            fake_images = generator(text_input)
            fake_loss = criterion(discriminator(fake_images.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            g_loss = criterion(discriminator(fake_images), real_labels)
            g_loss.backward()
            optimizer_G.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}] Batch {i}/{len(dataloader)} "
                      f"Discriminator Loss: {d_loss.item()} Generator Loss: {g_loss.item()}")

# Main
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = DreamScapeDataset(text_file='descriptions.txt', image_dir='images', transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    train(dataloader, generator, discriminator, optimizer_G, optimizer_D, criterion, device)

    # Save models
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

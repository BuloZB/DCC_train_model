import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50

# Definícia modelu UNet
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.model = fcn_resnet50(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)['out']

# Definícia transformácií pre obrázky a masky
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# Načítanie datasetu
train_data = datasets.ImageFolder(root='train_data_directory', transform=transform)
valid_data = datasets.ImageFolder(root='valid_data_directory', transform=transform)

train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=4, shuffle=False)

# Počet rôznych druhov defektov + 1 pre "žiadny defekt"
num_classes = 5

# Inicializácia modelu, stratovej funkcie a optimalizátora
model = UNet(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trénovanie modelu
num_epochs = 10

for epoch in range(num_epochs):
    for images, masks in train_loader:
        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Overenie modelu na validačných dátach
    total = 0
    correct = 0
    for images, masks in valid_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += masks.nelement()
        correct += (predicted == masks).sum().item()
    print(f'Epoch {epoch+1}, Accuracy: {correct / total}')

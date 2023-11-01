import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Define the path to your dataset
data_dir = "dataset"

# Define image transformations and create a dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to a common size
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
])

dataset = ImageFolder(root=data_dir, transform=transform)

# Define data loaders
batch_size = 16
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define a simple CNN model
class LungNet(nn.Module):
    def __init__(self):
        super(LungNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 112 * 112, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: Normal and Abnormal

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 112 * 112)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = LungNet()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(data_loader)}")

print("Training finished.")

# You can now use the trained model for classification.


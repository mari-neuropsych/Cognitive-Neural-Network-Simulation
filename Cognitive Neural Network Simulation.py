import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Download MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Define simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(28*28, 128)  # hidden layer
        self.output = nn.Linear(128, 10)     # output layer (10 digits)

    def forward(self, x):
        x = x.view(-1, 28*28)   # flatten input
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

# Initialize model, loss, optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (1 epoch for demonstration)
for images, labels in trainloader:
    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

print("Training done for 1 epoch")

# Test on first 10 images
test_images, test_labels = next(iter(trainloader))
with torch.no_grad():
    predictions = model(test_images[:10])
    predicted_labels = torch.argmax(predictions, dim=1)
    print("Predicted labels:", predicted_labels)
    print("True labels:     ", test_labels[:10])

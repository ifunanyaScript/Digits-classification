import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 100


# MNIST
train_data = torchvision.datasets.MNIST(root='.\data', train=True, 
                                        transform=transforms.ToTensor(), download=False)
test_data = torchvision.datasets.MNIST(root='.\data', train=False, 
                                        transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


examples = iter(train_loader)
samples, labels = examples.next()
# print(samples.shape, labels.shape)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
plt.show()

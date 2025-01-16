import torch
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
from sklearn.metrics import f1_score,accuracy_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config:
    input_Size=28*28
    hidden_size=100
    num_classes=10
    epoch=5
    lr=1e-4

train_Dataset=torchvision.datasets.MNIST(root="./data",train=True,transform=transforms.ToTensor(),download=True)
test_Dataset=torchvision.datasets.MNIST(root="./data",train=False,transform=transforms.ToTensor(),download=True)
train_loader=torch.utils.data.DataLoader(dataset=train_Dataset,batch_size=32,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_Dataset,batch_size=32,shuffle=False)
examples=iter(train_loader)
samples,target=next(examples)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
        self.loss=nn.CrossEntropyLoss()
        self.optimizer=torch.optim.Adam(self.parameters(),lr=1e-4)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
    def forward(self, x):
        out = self.l1(x)
        out=self.batch_norm1(out)
        out = self.relu(out)
        out = self.l2(out)
        
        # no activation and no softmax at the end
        return out

    def fit(self,train_loader):
        for epoch in range(Config.epoch):
            for i, (images, labels) in enumerate(train_loader):  
                # origin shape: [100, 1, 28, 28]
                # resized: [100, 784]
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs=self.forward(images)
                loss = self.loss(outputs, labels)
                
                # Backward and optimize
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if (i+1) % 100 == 0:
                    print (f'Epoch [{epoch+1}/{Config.epoch}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    def predict(self,x):
        
        with torch.no_grad():
           for i, (images, labels) in enumerate(x):  
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)
                outputs = self.forward(images)
                return outputs,labels

model=NeuralNet(28*28,256,10)
model = model.to(device)
model.fit(train_loader)
pred_val,labels=model.predict(test_loader)
print(torch.argmax(pred_val,dim=1).cpu().numpy())
print(labels) 

print(accuracy_score(torch.tensor(labels).cpu().numpy(),torch.argmax(pred_val,dim=1).cpu().numpy()))   
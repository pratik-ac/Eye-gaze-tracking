import numpy as np
import cv2 as cv
import os
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ConvNet_Yaxis(torch.nn.Module):
    def __init__(self):
        super(ConvNet_Yaxis, self).__init__()

        f2 = 8
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, f2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(f2),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(50 * 25 * f2, 200)
        self.fc2 = nn.Linear(200, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class ConvNet_Xaxis(torch.nn.Module):
    def __init__(self):
        super(ConvNet_Xaxis, self).__init__()

        f2 = 8
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, f2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(f2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(50 * 25 * f2, 200)
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def dataLoad(path, want=0):
    nameList = os.listdir(path)
    try:
        nameList.remove(".DS_Store")
    except:
        pass
    totalHolder = []
    dims = [1440, 900]
    for name in nameList:
        im = cv.cvtColor(cv.imread(path + "/" + name), cv.COLOR_BGR2GRAY)
        top = max([max(x) for x in im])
        totalHolder.append(((torch.tensor([[im]]).to(dtype=torch.float, device=device)) / top,
                            torch.tensor([[int((name.split("."))[want]) / dims[want]]]).to(dtype=torch.float, device=device)))
    return totalHolder

def evaluateModel(model, testSet, sidelen=1440):
    model.eval()
    errors = []
    criterion = nn.MSELoss()  # Define MSE loss
    for (im, label) in testSet:
        output = model(im)
        loss = criterion(output, label)  # Calculate MSE loss
        errors.append(loss.item())  # Append the loss value to errors list
    model.train()
    mean_error = sum(errors) / len(errors)  # Calculate the mean MSE loss
    return mean_error

trainingSet = dataLoad("eyes")
test = dataLoad("testeyes")
num_epochs = 50


bigTestX = []
bigTrainX = []

def trainModelX_axis():
    model = ConvNet_Xaxis().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    bestModel = model
    bestScore = 10000
    testscores = []
    trainscores = []

    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]:')
        np.random.shuffle(trainingSet)
        epoch_train_scores = []  # To store train scores for this epoch
        for i, (im, label) in enumerate(trainingSet):
            output = model(im)
            criterion = nn.MSELoss()  # Define MSE loss
            loss = criterion(output, label)  # Calculate MSE loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_scores.append(evaluateModel(model, [(im, label)]))  # Evaluate train score per sample
        trainSc = sum(epoch_train_scores) / len(epoch_train_scores)  # Average train score for this epoch
        testSc = evaluateModel(model, test)  # Evaluate test score for this epoch
        if testSc < bestScore:
            bestModel = copy.deepcopy(model)
            bestScore = testSc
        testscores.append(testSc)
        trainscores.append(trainSc)
        print(f'Training Score: {trainSc:.4f}')  # Print MSE for training
        print(f'Test Score: {testSc:.4f}')  # Print MSE for testing
    bigTestX.append(testscores)
    bigTrainX.append(trainscores)
    finalScoreX = evaluateModel(bestModel, test)
    print("Final Mean Squared Error ConvNet_Xaxis:", finalScoreX)
    if finalScoreX < 150:
        torch.save(model.state_dict(), f"xModels/model_x_epoch_{epoch+1}.plt") 

bigTestY = []
bigTrainY = []

def trainModelY_axis():
    model = ConvNet_Yaxis().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    bestModel = model
    bestScore = 10000
    testscores = []
    trainscores = []

    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]:')
        np.random.shuffle(trainingSet)
        epoch_train_scores = []  # To store train scores for this epoch
        for i, (im, label) in enumerate(trainingSet):
            output = model(im)
            criterion = nn.MSELoss()  # Define MSE loss
            loss = criterion(output, label)  # Calculate MSE loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_scores.append(evaluateModel(model, [(im, label)]))  # Evaluate train score per sample
        trainSc = sum(epoch_train_scores) / len(epoch_train_scores)  # Average train score for this epoch
        testSc = evaluateModel(model, test, sidelen=900)  # Evaluate test score for this epoch
        if testSc < bestScore:
            bestModel = copy.deepcopy(model)
            bestScore = testSc
        testscores.append(testSc)
        trainscores.append(trainSc)
        print(f'Training Score: {trainSc:.4f}')  # Print MSE for training
        print(f'Test Score: {testSc:.4f}')  # Print MSE for testing
    bigTestY.append(testscores)
    bigTrainY.append(trainscores)
    finalScoreY = evaluateModel(bestModel, test, sidelen=900)
    print("Final Mean Squared Error ConvNet_Yaxis:", finalScoreY)
    if finalScoreY < 150:
        torch.save(model.state_dict(), f"yModels/model_y_epoch_{epoch+1}.plt")

for i in range(1):
    trainModelX_axis()
    trainModelY_axis()


# Plotting X model scores
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), bigTrainX[0], label='ConvNet_Xaxis Train')
plt.plot(range(1, num_epochs + 1), bigTestX[0], label='ConvNet_Xaxis Test')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('ConvNet_Xaxis Model Training Progress')
plt.legend()
plt.show()


# Plotting Y model scores
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), bigTrainY[0], label='ConvNet_Yaxis Train')
plt.plot(range(1, num_epochs + 1), bigTestY[0], label='ConvNet_Yaxis Test')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('ConvNet_Yaxis Model Training Progress')
plt.legend()
plt.show()


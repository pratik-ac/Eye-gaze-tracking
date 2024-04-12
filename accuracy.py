import numpy as np
import cv2 as cv
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

def evaluateModel(model, testSet):
    model.eval()
    predictions = []
    labels = []
    criterion = nn.MSELoss()  # Define MSE loss
    for (im, label) in testSet:
        # Move input data to the same device as the model
        im = im.to(device)
        label = label.to(device)
        
        output = model(im)
        loss = criterion(output, label)  # Calculate MSE loss
        predictions.append(output.item())
        labels.append(label.item())
    model.train()
    
    # Binarize predictions
    threshold = 0.5  # Adjust the threshold as needed
    binary_predictions = np.where(np.array(predictions) >= threshold, 1, 0)

    # Convert labels to binary
    binary_labels = np.where(np.array(labels) >= threshold, 1, 0)

    # Calculate evaluation metrics
    accuracy = accuracy_score(binary_labels, binary_predictions)
    precision = precision_score(binary_labels, binary_predictions)
    recall = recall_score(binary_labels, binary_predictions)
    f1 = f1_score(binary_labels, binary_predictions)
    cm = confusion_matrix(binary_labels, binary_predictions)

    return accuracy, precision, recall, f1, cm


# Load the test dataset
test = dataLoad("testeyes")

# Load the trained model
ConvNetX = ConvNet_Xaxis()
ConvNetX.load_state_dict(torch.load("/home/von/Anti cheat/xModels/model_x_epoch_10.plt", map_location=device))  # Adjust the path as needed
ConvNetX.to(device)  # Move the model to the desired device
ConvNetX.eval()

ConvNetY = ConvNet_Yaxis()
ConvNetY.load_state_dict(torch.load("/home/von/Anti cheat/yModels/model_y_epoch_50.plt", map_location=device))  # Adjust the path as needed
ConvNetY.to(device)  # Move the model to the desired device
ConvNetY.eval()

# Evaluate the model
accuracy_x, precision_x, recall_x, f1_x, cm_x = evaluateModel(ConvNetX, test)
accuracy_y, precision_y, recall_y, f1_y, cm_y = evaluateModel(ConvNetY, test)

# Print the evaluation metrics
print("ConvNet_Xaxis:\n")
print("Accuracy:", accuracy_x)
print("Precision:", precision_x)
print("Recall:", recall_x)
print("F1 Score:", f1_x)
print("Confusion Matrix:\n", cm_x)
print("\n")
print("ConvNet_Yaxis:\n")
print("Accuracy:", accuracy_y)
print("Precision:", precision_y)
print("Recall:", recall_y)
print("F1 Score:", f1_y)
print("Confusion Matrix:\n", cm_y)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(cm_x, "CNN with one convolution layer for x-axis")
plot_confusion_matrix(cm_y, "CNN with one convolution layer for y-axis")

import time, sys
sys.path.append('../')  # add the parent directory of 'utils' to sys.path, whilst in beta build.
from pybci import PyBCI
from pybci.Configuration.EpochSettings import GlobalEpochSettings

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    
def PyTorchModel(x_train, x_test, y_train, y_test ):
    input_size = 15*8  # Number of input features
    hidden_size = 100  # Size of hidden layer
    num_classes = 3  # Number of output classes
    model = SimpleNN(input_size, hidden_size, num_classes)
    model.train()
    #criterion = torch.nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    epochs = 10
    train_data = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    model.eval()
    accuracy = 0
    with torch.no_grad():
        test_outputs = model(torch.Tensor(x_test))
        _, predicted = torch.max(test_outputs.data, 1)
        correct = (predicted == torch.Tensor(y_test)).sum().item()
        accuracy = correct / len(y_test)
    return accuracy, model # must return accuracy and model for pytorch!

generalEpochSettings = GlobalEpochSettings() # get general epoch time window settings (found in Configuration.EpochSettings.GlobalEpochSettings)
generalEpochSettings.windowLength = 1 # == tmax+tmin if generalEpochSettings.splitcheck is False, splits specified epochs in customEpochSettings

bci = PyBCI(minimumEpochsRequired = 4, globalEpochSettings = generalEpochSettings,  torchModel = PyTorchModel)

while not bci.connected:
    bci.Connect()
    time.sleep(1)

bci.TrainMode()
accuracy = 0
try:
    while(True):
        currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
        time.sleep(1) # wait for marker updates
        print("Markers received: " + str(currentMarkers) +" Class accuracy: " + str(accuracy))#, end="\r")
        if len(currentMarkers) > 1:  # check there is more then one marker type received
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired:
                classInfo = bci.CurrentClassifierInfo() # hangs if called too early
                accuracy = classInfo["accuracy"]
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+4:  
                bci.TestMode()
                break
    while True:
        markerGuess = bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
        guess = [key for key, value in currentMarkers.items() if value[0] == markerGuess]
        print("Current marker estimation: " + str(guess))#, end="\r")
        time.sleep(0.5)
except KeyboardInterrupt: # allow user to break while loop
    pass

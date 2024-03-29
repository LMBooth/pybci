from pybci import PyBCI
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU(inplace=True)  # In-place operation
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        if out.shape[0] > 1:  # Skip BatchNorm if batch size is 1
            out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        if out.shape[0] > 1:  # Skip BatchNorm if batch size is 1
            out = self.bn2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
def PyTorchModel(x_train, x_test, y_train, y_test):
    input_size = 2*8 # num of channels multipled by number of default features (rms and mean freq)
    hidden_size = 100
    num_classes = 4 # default in pseudodevice
    model = SimpleNN(input_size, hidden_size, num_classes)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    train_data = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train).long())
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True, drop_last=True)  # Drop last incomplete batch
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
        correct = (predicted == torch.Tensor(y_test).long()).sum().item()
        accuracy = correct / len(y_test)
    return accuracy, model

#@pytest.mark.timeout(300)  # Extended timeout to 5 minutes
def test_run_bci():
    bci = PyBCI(minimumEpochsRequired = 2, createPseudoDevice=True, torchModel=PyTorchModel)
    while not bci.connected:
        bci.Connect()
        time.sleep(1)
    bci.TrainMode()
    accuracy_achieved = False
    marker_received = False
    in_test_mode = False
    accuracy=None
    while True:
        currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
        time.sleep(0.5) # wait for marker updates
        #print("Markers received: " + str(currentMarkers) +" Accuracy: " + str(round(accuracy,2)), end="         \r")
        if len(currentMarkers) > 1:  # check there is more then one marker type received
            marker_received = True
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired:
                classInfo = bci.CurrentClassifierInfo() # hangs if called too early
                accuracy = classInfo["accuracy"]###
                if accuracy > 0:
                    # set to above 0 to show some accuracy was retruend from model
                    accuracy_achieved = True
                    bci.TestMode()
                    break
            #if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+4:
            #    break
    while True:
        markerGuess = bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
        print(markerGuess)
        #guess = [key for key, value in currentMarkers.items() if value[0] == markerGuess]
        in_test_mode = True
        time.sleep(1) 
        bci.StopThreads()
        break
        #print("Current marker estimation: " + str(guess), end="           \r")
    assert accuracy_achieved and marker_received and in_test_mode

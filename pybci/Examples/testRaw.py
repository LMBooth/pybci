import torch, time
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from pybci import PyBCI
import numpy as np
from pybci.Utils.Logger import Logger
num_chs = 3 # 8 channels re created in the PsuedoLSLGwnerator, but we drop 5 to save compute
sum_samps = 125 # sample rate is 250 in the PsuedoLSLGwnerator
num_classes = 3 # number of different triggers (can include baseline) sent, defines if we use softmax of binary
class ConvNet(nn.Module):
    def __init__(self, num_channels, num_samples, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.fc = nn.Linear(int(num_samples/2/2)*128, num_classes)  # Depending on your pooling and stride you might need to adjust the input size here
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    
def PyTorchModel(x_train, x_test, y_train, y_test ):
    model = ConvNet(num_chs, sum_samps, num_classes)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    train_data = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train).long())
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
        correct = (predicted == torch.Tensor(y_test).long()).sum().item()
        accuracy = correct / len(y_test)
    return accuracy, model # must return accuracy and model for pytorch!

class RawDecode():
    desired_length = 0
    def ProcessFeatures(self, epochData, sr, target): 
        d = epochData.T
        if self.desired_length == 0: # needed as windows may be differing sizes due to timestamp varience on LSL
            self.desired_length = d.shape[1]
        if d.shape[1] != self.desired_length:
            d = np.resize(d, (d.shape[0],self.desired_length))
        return d 

dropchs = [x for x in range(3,8)] # drop last 5 channels to save on compute time
streamChsDropDict={"sendTest":dropchs} #streamChsDropDict=streamChsDropDict,
streamCustomFeatureExtract = {"sendTest" : RawDecode()} # we select psuedolslgenerator example
bci = PyBCI(minimumEpochsRequired = 4, streamCustomFeatureExtract=streamCustomFeatureExtract, torchModel = PyTorchModel,streamChsDropDict=streamChsDropDict, loggingLevel = Logger.TIMING)
while not bci.connected:
    bci.Connect()
    time.sleep(1)

bci.TrainMode()
accuracy = 0
try:
    while(True):
        currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
        time.sleep(1) # wait for marker updates
        print("Markers received: " + str(currentMarkers) +" Class accuracy: " + str(accuracy), end="\r")
        if len(currentMarkers) > 1:  # check there is more then one marker type received
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired:
                classInfo = bci.CurrentClassifierInfo() # hangs if called too early
                accuracy = classInfo["accuracy"]
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+10:  
                featuresTargets = bci.CurrentFeaturesTargets() # when in test mode only y_pred returned
                print(featuresTargets["features"])
                print(featuresTargets["targets"])
                bci.TestMode()
                break
    while True:
        markerGuess = bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
        guess = [key for key, value in currentMarkers.items() if value[0] == markerGuess]
        print("Current marker estimation: " + str(guess), end="\r")
        time.sleep(0.1)
except KeyboardInterrupt: # allow user to break while loop
    pass

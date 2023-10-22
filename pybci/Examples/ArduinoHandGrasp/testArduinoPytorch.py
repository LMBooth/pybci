import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from pybci import PyBCI
from pybci.Configuration.EpochSettings import GlobalEpochSettings
from pybci.Utils.Logger import Logger
import numpy as np
# We control the arduino via lsl python script which is responsible for COM connection
from pylsl import StreamInfo, StreamOutlet
# Set up the LSL stream info
info = StreamInfo('MarkerHandGrasps', 'Markers', 1, 0, 'string', 'myuniquemarkerid2023')
# Create the outlet
outlet = StreamOutlet(info)
gs = GlobalEpochSettings()
gs.tmax = 2.5 # grab 1 second after marker
gs.tmin = -0.5 # grab 0.5 seconds after marker
gs.splitCheck = True # splits samples between tmin and tmax
gs.windowLength = 1  # 
gs.windowOverlap = 0.5 # windows overap by 50%, so for a total len

num_chs = 1 # 8 channels re created in the PsuedoLSLGwnerator, but we drop 7 as time series is computationally expensive!
num_samps = 192 # sample rate is 250 in the PsuedoLSLGwnerator
num_classes = 3 # number of different triggers (can include baseline) sent, defines if we use softmax of binary

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        # Only take the output from the final timestep
        out = out[:, -1, :]
        # Pass through the fully connected layer
        out = self.fc(out)
        return out
    
def PyTorchModel(x_train, x_test, y_train, y_test ):
    # Define the hyperparameters
    input_dim = num_chs
    hidden_dim = 128
    num_layers = 2
    model = LSTMNet(input_dim, hidden_dim, num_layers, num_classes)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 10
    # Reshape the input data to be [batch, sequence, feature]
    x_train = x_train.reshape(-1, num_samps, input_dim)
    x_test = x_test.reshape(-1, num_samps, input_dim)
    print(x_train.shape)
    print(y_train.shape)
    train_data = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train).long())
    train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
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

class RawDecode():
    desired_length = num_samps
    def ProcessFeatures(self, d, sr, target): 
        #print(epochData.T.shape)
        #d = epochData #.T
        print("rawdecode pre shape: ",d.shape)
        #if self.desired_length == 0: # needed as windows may be differing sizes due to timestamp varience on LSL
        #    self.desired_length = d.shape[1]
        if d.shape[0] != self.desired_length:
            #for ch in range(d.shape[0]):
            d = np.resize(d, (self.desired_length, 1))
        #print("rawdecode shape: ", d.shape)
        return d # we tranposeas using forloop for standardscalar normalises based on [channel,feature], whereas pull_chunk is [sample, channel]
        # for time series data we want to normalise each channel relative to itself
    
streamCustomFeatureExtract = {"ArduinoHandData":RawDecode()}
dataStreams = ["ArduinoHandData"]
bci = PyBCI(minimumEpochsRequired = 6, loggingLevel= Logger.INFO,torchModel=PyTorchModel, globalEpochSettings=gs,dataStreams=dataStreams, markerStream="TestMarkers", streamCustomFeatureExtract=streamCustomFeatureExtract)#, loggingLevel = Logger.NONE)
while not bci.connected: # check to see if lsl marker and datastream are available
    bci.Connect()
    time.sleep(1)

#ser.write(b'0\r')  # Convert the string to bytes and send 

bci.TrainMode() # now both marker and datastreams available start training on received epochs
accuracy = 0
try:
    while(True):
        currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
        time.sleep(0.1) # wait for marker updates
        print("Markers received: " + str(currentMarkers) +" Class accuracy: " + str(accuracy), end="\r")
        if len(currentMarkers) > 1:  # check there is more then one marker type received
            #print(bci.CurrentFeaturesTargets())
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired:
                classInfo = bci.CurrentClassifierInfo() # hangs if called too early
                accuracy = classInfo["accuracy"]
            if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+20:  
                #time.sleep(2)
                bci.TestMode()
                break
    while True:
        markerGuess = bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
        guess = [key for key, value in currentMarkers.items() if value[0] == markerGuess]
        print("Current marker estimation: " + str(guess), end="\r")
        time.sleep(0.5)
        if len(guess)>0:
            #outlet.push_sample([str(markerGuess)])
            if guess[0] == "open":
            #print("sending 0")
                outlet.push_sample(["0\r"])
                #ser.write(b"0\r")  # Convert the string to bytes and send 
            elif guess[0] == "fist":
                outlet.push_sample(["1\r"])
                #ser.write(b"1\r")  # Convert the string to bytes and send 
            elif guess[0] == "rock":
                outlet.push_sample(["2\r"])
                #ser.write(b"2\r")  # Convert the string to bytes and send 
            elif guess[0] == "peace":
                outlet.push_sample(["3\r"])
                #ser.write(b"3\r")  # Convert the string to bytes and send 
            elif guess[0] == "pinky":
                outlet.push_sample(["4\r"])
                #ser.write(b"4\r")  # Convert the string to bytes and send 
            elif guess[0] == "thumb":
                outlet.push_sample(["5\r"])
                #ser.write(b"5\r")  # Convert the string to bytes and send 

#["open", "fist", "rock"]#, "peace", "pinky"]
except KeyboardInterrupt: # allow user to break while loop
    pass
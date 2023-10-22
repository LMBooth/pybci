import time
import argparse
from pybci import PyBCI
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn


def main(createPseudoDevice=True, min_epochs_train=4, min_epochs_test=10, num_chs = 8, num_feats = 2, num_classes = 4):
    if createPseudoDevice:
        num_chs = 8 # 8 channels are created in the PseudoLSLGenerator
        num_feats = 2 # default is mean freq and rms to keep it simple
        num_classes = 4 # number of different triggers (can include baseline) sent, defines if we use softmax of binary

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
        input_size = num_feats*num_chs # num of channels multipled by number of default features (rms and mean freq)
        hidden_size = 100
        #num_classes = num_classes # default in pseudodevice
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

    bci = PyBCI(minimumEpochsRequired = min_epochs_train, createPseudoDevice=createPseudoDevice,  torchModel = PyTorchModel)
    while not bci.connected: # check to see if lsl marker and datastream are available
        bci.Connect()
        time.sleep(1)
    bci.TrainMode() # now both marker and datastreams available start training on received epochs
    accuracy = 0
    test = False
    try:
        while(True):
            if test is False:
                currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
                time.sleep(0.5) # wait for marker updates
                print("Markers received: " + str(currentMarkers) +" Accuracy: " + str(round(accuracy,2)), end="         \r")
                if len(currentMarkers) > 1:  # check there is more then one marker type received
                    if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired:
                        classInfo = bci.CurrentClassifierInfo() # hangs if called too early
                        accuracy = classInfo["accuracy"]
                    if min([currentMarkers[key][1] for key in currentMarkers]) > min_epochs_test:  
                        bci.TestMode()
                        break
            else:
                markerGuess = bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
                guess = [key for key, value in currentMarkers.items() if value[0] == markerGuess]
                print("Current marker estimation: " + str(guess), end="           \r")
                time.sleep(0.2)

        return None
    except KeyboardInterrupt: # allow user to break while loop
        print("\nLoop interrupted by user.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch neural network is used for model and pseudodevice generates 8 channels of 3 marker types and baseline. Similar to the testPytorch.py in the examples folder.")
    parser.add_argument("--createPseudoDevice", default=True, type=bool, help="Set to True or False to enable or disable pseudo device creation. pseudodevice generates 8 channels of 3 marker types and baseline.")
    parser.add_argument("--min_epochs_train", default=4, type=int, help='Minimum epochs to collect before model training commences, must be less than, min_epochs_test. If less than min_epochs_test defaults to min_epochs_test+1.')
    parser.add_argument("--min_epochs_test", default=14, type=int, help='Minimum epochs to collect before model testing commences, if less than min_epochs_test defaults to min_epochs_test+1.')
    parser.add_argument("--num_chs", default=8, type=int, help='Num of channels in data stream to configure tensorflow model, if PseudoDevice==True defaults to 8.')
    parser.add_argument("--num_classes", default=4, type=int, help='Num of classes in marker stream to configure tensorflow model, if PseudoDevice==True defaults to 4.')

    args = parser.parse_args()
    main(**vars(args))

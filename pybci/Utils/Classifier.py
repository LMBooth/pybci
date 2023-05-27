import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import tensorflow
import torch

from sklearn.model_selection import train_test_split
import numpy as np

class Classifier():
    classifierLibrary = "sklearn" # current default, should be none or somthing different?
    clf = svm.SVC(kernel = "rbf")#C=c, kernel=k, degree=d, gamma=g, coef0=c0, tol=t, max_iter=i)
    accuracy = 0
    model = None
    torchModel = None

    def __init__(self, clf = None, model = None, torchModel = None):
        super().__init__()
        if clf != None:
            self.clf = clf
        elif model != None:
            self.model = model
        elif torchModel != None:
            self.torchModel = model
        self.CheckClassifierLibrary()

    def CheckClassifierLibrary(self):
        if self.model != None: # maybe requires actual check for tensorflow model
            self.classifierLibrary = "tensor"
        elif self.torchModel != None: # maybe requires actual check for sklearn clf
            self.classifierLibrary = "pyTorch"
        elif self.clf != None: # maybe requires actual check for sklearn clf
            self.classifierLibrary = "sklearn"


    def TrainModel(self, features, targets):
        x_train, x_test, y_train, y_test = train_test_split(features, targets, shuffle = True, test_size=0.2)
        self.scaler = StandardScaler() # normalise our data (everything is a 0 or a 1 if you think about it, cheers georgey boy boole)
        self.scaler.fit(x_train)  # Compute the mean and standard deviation based on the training data
        x_train = self.scaler.transform(x_train)  # Scale the training data
        x_test = self.scaler.transform(x_test)  # Scale the test data
        if all(item == y_train[0] for item in y_train):
            pass
        else:
            #print(x_train, y_train)
            if self.classifierLibrary == "pytorch":
                self.accuracy, self.pymodel  = self.torchModel(x_train, x_test, y_train, y_test)
            elif self.classifierLibrary == "sklearn":
                self.clf.fit(x_train, y_train)
                y_predictions = self.clf.predict(x_test)
                self.accuracy = sklearn.metrics.accuracy_score(y_test, y_predictions)
            elif self.classifierLibrary == "tensor":
                self.model.fit(np.array(x_train), np.array(y_train)) # epochs and batch_size should be customisable
                self.loss, self.accuracy = self.model.evaluate(np.array(x_test), np.array(y_test))
            else:
                # no classifier library selected, print debug?
                pass

    def TestModel(self, x):
        x = self.scaler.transform([x])[0]  # Scale the test data
        if self.classifierLibrary == "sklearn":
            x = np.expand_dims(x, axis=0)
            return self.clf.predict(x)
        elif self.classifierLibrary == "tensor":
            x = np.expand_dims(x, axis=0)
            predictions = self.model.predict(x)
            if len (predictions[0]) == 1: # assume binary classification
                return 1 if predictions[0] > 0.5 else 0
            else:    # assume multi-classification
                return np.argmax(predictions[0])
        elif self.classifierLibrary == "pyTorch":
            x = torch.Tensor(np.expand_dims(x, axis=0))
            self.pymodel.eval()
            with torch.no_grad():
                predictions = self.pymodel(x)
                if len (predictions[0]) == 1: # assume binary classification
                    return 1 if predictions[0] > 0.5 else 0
                else:    # assume multi-classification
                    return torch.argmax(predictions).item()

        else:
            print("no classifier library selected")
            # no classifier library selected, print debug?
            pass

'''
    def UpdateModel(self, featuresSingle, target): 
        # function currently not used, may be redundant, means thread function hold feature and target variables and passes reference to here, 
        # would be better to hold in classifier class?
        featuresSingle = np.where(np.isnan(featuresSingle), 0, featuresSingle)
        if (len(np.array(self.features).shape) ==3):
            features  = np.array(features).reshape(np.array(features).shape[0], -1)
        self.features = np.vstack([self.features, featuresSingle])
        self.targets = np.hstack([self.targets, target])
        if self.classifierLibrary == "sklearn":
            # Update the model with new data using partial_fit
            self.clf.fit(self.features, self.targets) #, classes=np.unique(target))
            self.accuracy = self.clf.score(self.x_test, self.y_test)
        elif self.classifierLibrary == "tensor":
            self.model.fit(featuresSingle, target, epochs=1, batch_size=32)
            self.loss, self.accuracy = self.model.evaluate(self.x_test, self.y_test)
        else:
            # no classifier library selected, print debug?
            pass
'''
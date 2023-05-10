import sklearn
import tensorflow
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm

class Classifier():
    classifierLibrary = "sklearn" # current default, should be none or somthing different?
    clf = svm.SVC(kernel = "rbf")#C=c, kernel=k, degree=d, gamma=g, coef0=c0, tol=t, max_iter=i)
    accuracy = 0
    def __init__(self, clf = None, model = None):
        super().__init__()
        if clf != None:
            self.clf = clf
        elif model != None:
            self.model = model
        self.CheckClassifierLibrary()

    def CheckClassifierLibrary(self):
        if self.clf != None: # maybe requires actual check for sklearn clf
            self.classifierLibrary = "sklearn"
        elif self.model != None: # maybe requires actual check for tensorflow model
            self.classifierLibrary = "tensor"

    def TrainModel(self, features, targets):
        if (len(np.array(features).shape) ==3):
            features  = np.array(features).reshape(np.array(features).shape[0], -1)
        else:
            features = np.array(features)
        features = np.where(np.isnan(features), 0, features)
        #self.targets = targets
        x_train, x_test, y_train, y_test = train_test_split(features, targets, shuffle = True, test_size=0.2)
        #print("training model")
        if self.classifierLibrary == "sklearn":
            if all(item == y_train[0] for item in y_train):
                pass
            else:
                self.clf.fit(x_train, y_train)
                y_predictions = self.clf.predict(x_test)
                print(y_predictions)
                self.accuracy = sklearn.metrics.accuracy_score(y_test, y_predictions)
                print("Classification accuracy:" +str(self.accuracy))
        elif self.classifierLibrary == "tensor":
            if all(item == y_train[0] for item in y_train):
                pass
            else:
                self.model.fit(x_train, y_train, epochs=1, batch_size=32)
                self.loss, self.accuracy = self.model.evaluate(x_test, y_test)
        else:
            # no classifier library selected, print debug?
            pass

    def TestModel(self, x):
        if (len(np.array(x).shape) ==3):
            x = np.array(x).reshape(np.array(x).shape[0], -1)
        else:
            x = np.array(x)
        print(x.shape)
        if self.classifierLibrary == "sklearn":
            y_pred = self.clf.predict(x)
            print("we predict it's: "+str(y_pred))
        elif self.classifierLibrary == "tensor":
            # Predict the class labels for the test data
            y_pred = self.model.predict(x)
            print("we predict it's: "+str(y_pred))
            # Convert the predicted probabilities to class labels
            y_pred_classes = y_pred.argmax(axis=-1)
            print("if class label: "+str(y_pred_classes))
        else:
            # no classifier library selected, print debug?
            pass


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

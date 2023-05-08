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

    def CompileModel(self, features, targets):
        x_train, x_test, y_train, y_test = train_test_split(features, targets, shuffle = True, test_size=0.2)
        self.x_test, self.y_test = x_test, y_test # get calidation sets
        if self.classifierLibrary == "sklearn":
            self.clf.fit(x_train, y_train)
            y_predictions = self.clf.predict(x_test)
            self.accuracy = sklearn.metrics.accuracy_score(y_test, y_predictions)
        elif self.classifierLibrary == "tensor":
            self.model.fit(features, targets, epochs=1, batch_size=32)
            self.loss, self.accuracy = self.model.evaluate(self.x_test, self.y_test)
        else:
            # no classifier library selected, print debug?
            pass

    def UpdateModel(self, featuresSingle, target):
        if self.classifierLibrary == "sklearn":
            # Update the model with new data using partial_fit
            self.clf.partial_fit(featuresSingle, target, classes=np.unique(target))
            self.accuracy = self.clf.score(self.x_test, self.y_test)
        elif self.classifierLibrary == "tensor":
            self.model.fit(featuresSingle, target, epochs=1, batch_size=32)
            self.loss, self.accuracy = self.model.evaluate(self.x_test, self.y_test)
        else:
            # no classifier library selected, print debug?
            pass
    
    def TestModel(self, x):
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


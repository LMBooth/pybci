import argparse
from Tests.testSimple import main as testSimple
from Tests.testSklearn import main as testSklearn
from Tests.testTensorflow import main as testTensorflow
from Tests.testPyTorch import main as testPyTorch

def main():
    parser = argparse.ArgumentParser(description='PyBCI: A Brain-Computer Interface package.')
    
    subparsers = parser.add_subparsers(title='Commands', description='Available example commands')
    
    testSimple_parser = subparsers.add_parser('testSimple', help='Runs simple setup where sklearn support-vector-machine is used for model and pseudodevice generates 8 channels of 3 marker types and baseline. Similar to the testSimple.py in the examples folder.')
    testSimple_parser.add_argument('--createPseudoDevice',default=True,  type=bool, help='Set to True or False to enable or disable pseudo device creation.')
    testSimple_parser.add_argument('--min_epochs_train', default=4, type=int, help='Minimum epochs to collect before model training commences, must be less than, min_epochs_test. If less than min_epochs_test defaults to min_epochs_test+1.')
    testSimple_parser.add_argument('--min_epochs_test', default=10, type=int, help='Minimum epochs to collect before model testing commences, if less than min_epochs_test defaults to min_epochs_test+1.')
    testSimple_parser.set_defaults(func=testSimple)
    
    testSklearn_parser = subparsers.add_parser('testSklearn', help='Sklearn multi-layer perceptron is used for model and pseudodevice generates 8 channels of 3 marker types and baseline. Similar to the testSklearn.py in the examples folder.')
    testSklearn_parser.add_argument('--createPseudoDevice',default=True, type=bool, help='Set to True or False to enable or disable pseudo device creation.')
    testSklearn_parser.add_argument('--min_epochs_train', default=4,type=int, help='Minimum epochs to collect before model training commences, must be less than, min_epochs_test. If less than min_epochs_test defaults to min_epochs_test+1.')
    testSklearn_parser.add_argument('--min_epochs_test', default=10,type=int, help='Minimum epochs to collect before model testing commences, if less than min_epochs_test defaults to min_epochs_test+1.')
    testSklearn_parser.set_defaults(func=testSklearn)
    
    testTensorflow_parser = subparsers.add_parser('testTensorflow', help='Tensorflow GRU is used for model and pseudodevice generates 8 channels of 3 marker types and baseline. Similar to the testTensorflow.py in the examples folder.')
    testTensorflow_parser.add_argument("--createPseudoDevice", default=True, type=bool, help="Set to True or False to enable or disable pseudo device creation.")
    testTensorflow_parser.add_argument("--min_epochs_train", default=4, type=int, help='Minimum epochs to collect before model training commences, must be less than, min_epochs_test. If less than min_epochs_test defaults to min_epochs_test+1.')
    testTensorflow_parser.add_argument("--min_epochs_test", default=10, type=int, help='Minimum epochs to collect before model testing commences, if less than min_epochs_test defaults to min_epochs_test+1.')
    testTensorflow_parser.add_argument("--num_chs", default=8, type=int, help='Num of channels in data stream to configure tensorflow model, if PseudoDevice==True defaults to 8.')
    testTensorflow_parser.add_argument("--num_classes", default=4, type=int, help='Num of classes in marker stream to configure tensorflow model, if PseudoDevice==True defaults to 4.')
    testTensorflow_parser.set_defaults(func=testTensorflow)

    testPyTorch_parser = subparsers.add_parser('testPyTorch', help='PyTorch neural network is used for model and pseudodevice generates 8 channels of 3 marker types and baseline. Similar to the testPytorch.py in the examples folder.')
    testPyTorch_parser.add_argument("--createPseudoDevice", default=True, type=bool, help="Set to True or False to enable or disable pseudo device creation.")
    testPyTorch_parser.add_argument("--min_epochs_train", default=4, type=int, help='Minimum epochs to collect before model training commences, must be less than, min_epochs_test. If less than min_epochs_test defaults to min_epochs_test+1.')
    testPyTorch_parser.add_argument("--min_epochs_test", default=10, type=int, help='Minimum epochs to collect before model testing commences, if less than min_epochs_test defaults to min_epochs_test+1.')
    testPyTorch_parser.add_argument("--num_chs", default=8, type=int, help='Num of channels in data stream to configure tensorflow model, if PseudoDevice==True defaults to 8.')
    testPyTorch_parser.add_argument("--num_classes", default=4, type=int, help='Num of classes in marker stream to configure tensorflow model, if PseudoDevice==True defaults to 4.')
    testPyTorch_parser.set_defaults(func=testPyTorch)

    

    args = parser.parse_args()
    args.func(**vars(args))

if __name__ == '__main__':
    main()

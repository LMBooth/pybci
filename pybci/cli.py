import argparse
import time


def import_and_run_testSimple(**kwargs):
    from .CliTests.testSimple import main as testSimple
    testSimple(**kwargs)

def import_and_run_testSklearn(**kwargs):
    from .CliTests.testSklearn import main as testSklearn
    testSklearn(**kwargs)

def import_and_run_testTensorflow(**kwargs):
    from .CliTests.testTensorflow import main as testTensorflow
    testTensorflow(**kwargs)

def import_and_run_testPyTorch(**kwargs):
    from .CliTests.testPyTorch import main as testPyTorch
    testPyTorch(**kwargs)

def RunPseudo():
    from .Utils.PseudoDevice import PseudoDeviceController
    pseudoDevice = PseudoDeviceController()
    pseudoDevice.BeginStreaming()
    while True:
        time.sleep(1)

def main():
    parser = argparse.ArgumentParser(description='PyBCI: A Brain-Computer Interface package. Visit https://pybci.readthedocs.io/en/latest/ for more information!')
    
    subparsers = parser.add_subparsers(title='Commands', description='Available example commands')
    
    testSimple_parser = subparsers.add_parser('testSimple', help='Runs simple setup where sklearn support-vector-machine is used for model and pseudodevice generates 8 channels of 3 marker types and baseline. Similar to the testSimple.py in the examples folder.')
    testSimple_parser.add_argument('--createPseudoDevice',default=True,  type=bool, help='Set to True or False to enable or disable pseudo device creation. pseudodevice generates 8 channels of 3 marker types and baseline.')
    testSimple_parser.add_argument('--min_epochs_train', default=4, type=int, help='Minimum epochs to collect before model training commences, must be less than, min_epochs_test. If less than min_epochs_test defaults to min_epochs_test+1.')
    testSimple_parser.add_argument('--min_epochs_test', default=14, type=int, help='Minimum epochs to collect before model testing commences, if less than min_epochs_test defaults to min_epochs_test+1.')
    testSimple_parser.add_argument("--timeout", default=None, type=int, help="Timeout in seconds for the script to automatically stop.")

    #testSimple_parser.set_defaults(func=testSimple)
    
    testSklearn_parser = subparsers.add_parser('testSklearn', help='Sklearn multi-layer perceptron is used for model and pseudodevice generates 8 channels of 3 marker types and baseline. Similar to the testSklearn.py in the examples folder.')
    testSklearn_parser.add_argument('--createPseudoDevice',default=True, type=bool, help='Set to True or False to enable or disable pseudo device creation. pseudodevice generates 8 channels of 3 marker types and baseline.')
    testSklearn_parser.add_argument('--min_epochs_train', default=4,type=int, help='Minimum epochs to collect before model training commences, must be less than, min_epochs_test. If less than min_epochs_test defaults to min_epochs_test+1.')
    testSklearn_parser.add_argument('--min_epochs_test', default=14,type=int, help='Minimum epochs to collect before model testing commences, if less than min_epochs_test defaults to min_epochs_test+1.')
    testSklearn_parser.add_argument("--timeout", default=None, type=int, help="Timeout in seconds for the script to automatically stop.")

    #testSklearn_parser.set_defaults(func=testSklearn)
    
    testTensorflow_parser = subparsers.add_parser('testTensorflow', help='Tensorflow GRU is used for model and pseudodevice generates 8 channels of 3 marker types and baseline. Similar to the testTensorflow.py in the examples folder.')
    testTensorflow_parser.add_argument("--createPseudoDevice", default=True, type=bool, help="Set to True or False to enable or disable pseudo device creation. pseudodevice generates 8 channels of 3 marker types and baseline.")
    testTensorflow_parser.add_argument("--min_epochs_train", default=4, type=int, help='Minimum epochs to collect before model training commences, must be less than, min_epochs_test. If less than min_epochs_test defaults to min_epochs_test+1.')
    testTensorflow_parser.add_argument("--min_epochs_test", default=14, type=int, help='Minimum epochs to collect before model testing commences, if less than min_epochs_test defaults to min_epochs_test+1.')
    testTensorflow_parser.add_argument("--num_chs", default=8, type=int, help='Num of channels in data stream to configure tensorflow model, if PseudoDevice==True defaults to 8.')
    testTensorflow_parser.add_argument("--num_classes", default=4, type=int, help='Num of classes in marker stream to configure tensorflow model, if PseudoDevice==True defaults to 4.')
    testTensorflow_parser.add_argument("--timeout", default=None, type=int, help="Timeout in seconds for the script to automatically stop.")

    #testTensorflow_parser.set_defaults(func=testTensorflow)

    testPyTorch_parser = subparsers.add_parser('testPyTorch', help='PyTorch neural network is used for model. Similar to the testPytorch.py in the examples folder.')
    testPyTorch_parser.add_argument("--createPseudoDevice", default=True, type=bool, help="Set to True or False to enable or disable pseudo device creation. pseudodevice generates 8 channels of 3 marker types and baseline.")
    testPyTorch_parser.add_argument("--min_epochs_train", default=4, type=int, help='Minimum epochs to collect before model training commences, must be less than, min_epochs_test. If less than min_epochs_test defaults to min_epochs_test+1.')
    testPyTorch_parser.add_argument("--min_epochs_test", default=14, type=int, help='Minimum epochs to collect before model testing commences, if less than min_epochs_test defaults to min_epochs_test+1.')
    testPyTorch_parser.add_argument("--num_chs", default=8, type=int, help='Num of channels in data stream to configure tensorflow model, if PseudoDevice==True defaults to 8.')
    testPyTorch_parser.add_argument("--num_classes", default=4, type=int, help='Num of classes in marker stream to configure tensorflow model, if PseudoDevice==True defaults to 4.')
    testPyTorch_parser.add_argument("--timeout", default=None, type=int, help="Timeout in seconds for the script to automatically stop.")

    #testPyTorch_parser.set_defaults(func=testPyTorch)


    testPseudo = subparsers.add_parser('createPseudoStreams', help='Creates basic Pseudo Device data and marker Lab Streaming Layer (LSL) streams.')
    testPseudo.set_defaults(func=RunPseudo)

    testSimple_parser.set_defaults(func=import_and_run_testSimple)
    testSklearn_parser.set_defaults(func=import_and_run_testSklearn)
    testTensorflow_parser.set_defaults(func=import_and_run_testTensorflow)
    testPyTorch_parser.set_defaults(func=import_and_run_testPyTorch)

    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
    else:
        arg_dict = vars(args)
        func = arg_dict.pop('func')  # Remove 'func' and store the actual function
        func(**arg_dict)  # Call the function with the remaining arguments


if __name__ == '__main__':
    main()

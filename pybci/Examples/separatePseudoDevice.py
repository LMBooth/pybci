import time
from pybci.Utils.PseudoDevice import PseudoDeviceController

# Initialize pd globally
pd = None

def run_pseudo():
    global pd
    pd = PseudoDeviceController(execution_mode="process")
    pd.BeginStreaming()

if __name__ == '__main__':
    try:
        run_pseudo()
        while True:  # Loop indefinitely
            time.sleep(0.5)  # Sleep to prevent this loop from consuming too much CPU
    except KeyboardInterrupt:
        # Safely handle pd to ensure it's not None and StopStreaming method is available
        if pd and hasattr(pd, 'StopStreaming'):
            pd.StopStreaming()
        print("KeyboardInterrupt has been caught. Stopping the script.")

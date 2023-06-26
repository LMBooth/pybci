import tkinter as tk
import pylsl

# customisable variables!
stimuli = ["Both Open", "Right Open", "Left Open"]
stimuliTime = [3000, 3000, 3000]
stimuliCount = [8, 8, 8]

markerStreamName = "TestMarkers" # should be targetted with pybci
streamType = 'Markers'

class App:
    def __init__(self, root):
        markerInfo = pylsl.StreamInfo(markerStreamName, streamType, 1, 0, 'string', 'Dev') # creates lsl marker info
        self.markerOutlet = pylsl.StreamOutlet(markerInfo) # creates lsl marker outlet from marker info
        self.originalstimuliCount = stimuliCount
        self.root = root
        self.root.state("zoomed")  # Maximize the window
        self.root.grid_rowconfigure(0, weight=1)  # Configure row 0 to expand vertically
        self.root.grid_columnconfigure(0, weight=1)  # Configure column 0 to expand horizontally
        self.root.grid_columnconfigure(1, weight=1)  # Configure column 1 to expand horizontally

        self.label = tk.Label(root, text="", font=("Helvetica", 24))
        self.label.grid(row=0, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")

        self.button = tk.Button(root, text="Start", command=self.toggle_iteration, font=("Helvetica", 18))
        self.button.grid(row=1, column=0,columnspan=2, padx=20, pady=20, sticky="nsew")
        #self.custom_button = tk.Button(root, text="Start Testing", command=self.custom_function, font=("Helvetica", 18))
        #self.custom_button.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")
        self.close_button = tk.Button(root, text="Close", command=self.root.destroy, font=("Helvetica", 18))
        self.close_button.grid(row=2, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")

        self.index = 0
        self.iterating = False  # Variable to track the iteration state
        self.after_id = None  # Variable to store the after() call ID

    def toggle_iteration(self):
        if not self.iterating:
            self.iterating = True
            self.button.configure(text="Stop")
            self.next_stimulus()
        else:
            self.iterating = False
            self.button.configure(text="Start")
            if self.after_id is not None:
                self.root.after_cancel(self.after_id)
                self.after_id = None
                
    def next_stimulus(self):
        if self.iterating:
            if len(stimuli) == 0:
                self.label['text'] = "Finished"
            else:
                self.label['text'] = stimuli[self.index]
                self.markerOutlet.push_sample([stimuli[self.index]])
                print("sent marker")
                self.after_id = self.root.after(stimuliTime[self.index], self.next_stimulus)
                stimuliCount[self.index] -= 1
                if stimuliCount[self.index] == 0:
                    self.remove_stimulus(self.index)
                else:
                    self.index = (self.index + 1) % len(stimuli)  # Increment index and wrap around when it reaches the end

    def remove_stimulus(self, index):
        del stimuli[index]
        del stimuliTime[index]
        del stimuliCount[index]
        if len(stimuli) == 0:
            self.iterating = False
            self.button.configure(text="Start")
            if self.after_id is not None:
                self.root.after_cancel(self.after_id)
                self.after_id = None
    
    #def custom_function(self):
    #    # Define your custom function here
    #    print("Custom function called")

root = tk.Tk()
app = App(root)
root.mainloop()

class Logger:
    INFO = "INFO"
    WARNING = "WARNING"
    NONE = "NONE"
    TIMING = "TIMING"

    def __init__(self, level=INFO, log_queue=None):
        self.queue = log_queue
        self.level = level
        self.check_level(level)
        #print(self.level)
    def set_level(self, level):
        self.level = level
        self.check_level(level)

    def check_level(self,level):
        if level != self.WARNING and level != self.INFO and level != self.NONE and level != self.TIMING :
            print("PyBCI: [INFO] - Invalid or no log level selected, defaulting to info. (options: info, warning, none)")
            level = self.INFO
            self.level = level

    def log(self, level, message):
        if self.level == self.NONE:
            return None
        if level == self.INFO:
            if self.level != self.NONE and self.level != self.WARNING:
                print('PyBCI: [INFO] -' + message)
        elif level == self.WARNING:
            if self.level != self.NONE:
                print('PyBCI: [WARNING] -' + message)
        elif level == self.TIMING:
            if self.level == self.TIMING:
                print('PyBCI: [TIMING] -' + message)

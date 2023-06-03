class Logger:
    INFO = "info"
    WARNING = "warning"
    NONE = "none"
    TIMING = "timing"

    def __init__(self, level=INFO):
        self.level = level
        self.check_level(level)

    def set_level(self, level):
        self.level = level
        self.check_level(level)

    def check_level(self,level):
        if level != self.WARNING and level != self.INFO and level != self.NONE and level != self.TIMING :
            print("PyBCI: [INFO] - Invalid or no log level selected, defaulting to info. (options: info, warning, none)")
            level = self.INFO
            self.level = level

    def log(self, level, message):
        if self.level == 'none':
            return None
        if level == 'info':
            if self.level != 'none' and self.level != 'warning':
                print('PyBCI: [INFO] -' + message)
        elif level == 'warning':
            if self.level != 'none':
                print('PyBCI: [WARNING] -' + message)
        elif level == 'timing':
            if self.level == 'timing':
                print('PyBCI: [TIMING] -' + message)
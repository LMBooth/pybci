class Logger:
    INFO = "info"
    WARNING = "warning"
    NONE = "none"

    def __init__(self, level=INFO):
        self.level = level
        self.check_level(level)

    def set_level(self, level):
        self.level = level
        self.check_level(level)

    def check_level(self,level):
        if level != self.WARNING or level != self.INFO or level != self.NONE:
            print("PyBCI: [WARNING] - Invalid log level selected, defaulted to info. (options: info, warning, none)")
            level = self.INFO
            self.level = level

    def log(self, level, message):
        if self.level == 'none':
            return

        if level == 'info':
            if self.level != 'none':
                print('PyBCI: [INFO] -' + message)
        elif level == 'warning':
            if self.level == 'warning':
                print('PyBCI: [WARNING] -' + message)
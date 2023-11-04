        if createPseudoDevice:
            #current_os = get_os()
            #if current_os == "Windows":
            if isinstance(pseudoDeviceController,PseudoDeviceController):
                pseudoDeviceController.BeginStreaming()
            else:
                pseudoDeviceController = PseudoDeviceController()
                pseudoDeviceController.BeginStreaming()
            self.pseudoDeviceController = pseudoDeviceController
            #elif current_os == "macOS":
            #    self.pseudoDeviceController = pseudoDeviceController

            #elif current_os == "Linux":
            #    self.pseudoDeviceController = pseudoDeviceController

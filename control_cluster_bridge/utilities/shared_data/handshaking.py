from SharsorIPCpp.PySharsorIPC import StringTensorServer, StringTensorClient
from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedDataView
from SharsorIPCpp.PySharsor.wrappers.shared_tensor_dict import SharedTensorDict
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import dtype as sharsor_dtype 
from SharsorIPCpp.PySharsorIPC import Journal

from control_cluster_bridge.utilities.shared_data.abstractions import SharedDataBase

from typing import List

class HandShaker(SharedDataBase):

    def __init__(self,
            shared_data_list: List):

        self._handshake_done = False

        self.shared_data_list = shared_data_list

        self._check_shared_data()

        self._is_runnning = False

    def _check_shared_data(self):

        # Check if input_list is a list
        if not isinstance(self.shared_data_list, list):

            exception = f"Input should be a list!!"

            Journal.log(self.__class__.__name__,
                "_check_shared_data",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        
        # Check if all elements in the list are SharedDataBase
        if not all(isinstance(element, SharedDataBase) for element in self.shared_data_list):

            exception = f"Input should be a list of SharedDataBase objects"

            Journal.log(self.__class__.__name__,
                "_check_shared_data",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
    
    def run(self):

        # for shared_data in self.shared_data_list:
            
        #     if not shared_data.is_running():

        #         shared_data.run()
        

        self._is_runnning = True

    def close(self):

        pass
    
    def is_running(self):

        return self._is_runnning


from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Producer, Consumer


class RemoteTriggererSrvr(Producer):

    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V0,
            force_reconnection: bool = False):

        super().__init__(namespace=namespace,
            basename="RemoteRHC",
            verbose=verbose,
            vlevel=vlevel,
            force_reconnection=force_reconnection)

class RemoteTriggererClnt(Consumer):

    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V0):

        super().__init__(namespace=namespace,
            basename="RemoteRHC",
            verbose=verbose,
            vlevel=vlevel)
 
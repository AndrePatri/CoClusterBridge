from abc import ABC, abstractmethod

class SharedDataBase(ABC):

    @abstractmethod
    def run(self):

        pass

    @abstractmethod
    def close(self):

        pass

    @abstractmethod
    def is_running(self):

        pass

    @abstractmethod
    def get_shared_mem(self):

        pass
    
def is_shared_data_child(cls):

    return issubclass(cls,
             SharedDataBase)
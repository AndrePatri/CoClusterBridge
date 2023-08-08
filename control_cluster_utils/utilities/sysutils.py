import os

class PathsGetter:

    def __init__(self):
        
        self.PACKAGE_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

        self.PIPES_CONFIGPATH = os.path.join(self.PACKAGE_ROOT_DIR, 
                                            'config', 
                                            'pipes', 
                                            'pipes_config.yaml')
        
        self.SHARED_MEM_CONFIGPATH = os.path.join(self.PACKAGE_ROOT_DIR, 
                                            'config', 
                                            'shared_mem', 
                                            'shared_mem_config.yaml')
        
        self.CONTROLLERS_PATH = os.path.join(self.PACKAGE_ROOT_DIR, 
                                            'controllers')
        
        self.CLUSTER_SRV_PATH = os.path.join(self.PACKAGE_ROOT_DIR, 
                                            'cluster_server')

        self.CLUSTER_CLT_PATH = os.path.join(self.PACKAGE_ROOT_DIR, 
                                            'cluster_client')

        self.UTILS_PATH = os.path.join(self.PACKAGE_ROOT_DIR, 
                                            'utilities')

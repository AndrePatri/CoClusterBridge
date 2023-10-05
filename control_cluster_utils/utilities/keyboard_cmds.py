import torch

from pynput import keyboard

from control_cluster_utils.utilities.rhc_defs import RhcTaskRefs
from control_cluster_utils.utilities.shared_mem import SharedMemClient
from control_cluster_utils.utilities.shared_mem import SharedMemSrvr
from control_cluster_utils.utilities.defs import cluster_size_name
from control_cluster_utils.utilities.defs import n_contacts_name
from control_cluster_utils.utilities.defs import launch_keybrd_cmds_flagname
from control_cluster_utils.utilities.defs import env_selector_name

class RhcRefsFromKeyboard:

    def __init__(self, 
                namespace: str, 
                verbose = False):

        self._verbose = verbose

        self.namespace = namespace

        self._terminated = False

        self.contacts = None 
        
        self.enable_heightchange = False

        self.cluster_size = -1
        self.n_contacts = -1

        self.cluster_size_clnt = None

        self.rhc_task_refs = []

        self._init_shared_data()

    def _init_shared_data(self):

        wait_amount = 0.05
        
        self.cluster_size_clnt = SharedMemClient(name=cluster_size_name(), 
                                    namespace=self.namespace,
                                    dtype=torch.int64, 
                                    wait_amount=wait_amount, 
                                    verbose=self._verbose)
        self.cluster_size_clnt.attach()
        
        self.n_contacts_clnt = SharedMemClient(name=n_contacts_name(), 
                                    namespace=self.namespace, 
                                    dtype=torch.int64, 
                                    wait_amount=wait_amount, 
                                    verbose=self._verbose)
        self.n_contacts_clnt.attach()

        self.launch_keyboard_cmds = SharedMemSrvr(n_rows=1, 
                                        n_cols=1, 
                                        name=launch_keybrd_cmds_flagname(), 
                                        namespace=self.namespace,
                                        dtype=torch.bool)
        self.launch_keyboard_cmds.start()
        self.launch_keyboard_cmds.reset_bool(False)

        self.env_index = SharedMemClient(name=env_selector_name(), 
                                        namespace=self.namespace, 
                                        dtype=torch.int64, 
                                        wait_amount=wait_amount, 
                                        verbose=self._verbose)
        self.env_index.attach()

        # while self.launch_keyboard_cmds.get_clients_count() != 1:

        #     print("[RhcRefsFromKeyboard]" + self._init_shared_data.__name__ + ":"\
        #         " waiting for exactly one client to connect")

        self.cluster_size = self.cluster_size_clnt.tensor_view[0, 0].item()
        self.n_contacts = self.n_contacts_clnt.tensor_view[0, 0].item()
        
        self.cluster_idx = self.env_index.tensor_view[0, 0]
        
        self.contacts = torch.tensor([[True] * self.n_contacts], 
                        dtype=torch.float32)
        
        self._init_shared_task_data()

    def _init_shared_task_data(self):

        # view of rhc references
        for i in range(0, self.cluster_size):

            self.rhc_task_refs.append(RhcTaskRefs( 
                n_contacts=self.n_contacts,
                index=i,
                q_remapping=None,
                namespace=self.namespace,
                dtype=torch.float32, 
                verbose=self._verbose))
        
    def __del__(self):

        if not self._terminated:

            self._terminate()
    
    def _terminate(self):
        
        self._terminated = True

        self.__del__()
    
    def _update(self):

        self.cluster_idx = int(self.env_index.tensor_view[0, 0])

    def _set_cmds(self):

        self.rhc_task_refs[self.cluster_idx].phase_id.set_contacts(
                                self.contacts)

    def _on_press(self, key):

        if self.launch_keyboard_cmds.all():
            
            self._update() # updates  data like
            # current cluster index

            if hasattr(key, 'char'):
                
                print('Key {0} pressed.'.format(key.char))
                
                if key.char == "9":
                    
                    self.contacts[0, 0] = False

                if key.char == "7":
                    
                    self.contacts[0, 1] = False

                if key.char == "1":
                    
                    self.contacts[0, 2] = False

                if key.char == "3":
                    
                    self.contacts[0, 3] = False

            self._set_cmds()

    def _on_release(self, key):

        if self.launch_keyboard_cmds.all():
            
            self._update() # updates  data like
            # current cluster index
            
            if hasattr(key, 'char'):
                
                print('Key {0} released.'.format(key.char))

                if key.char == "9":
                    
                    self.contacts[0, 0] = True

                if key.char == "7":
                    
                    self.contacts[0, 1] = True

                if key.char == "1":
                    
                    self.contacts[0, 2] = True

                if key.char == "3":
                    
                    self.contacts[0, 3] = True

                if key == keyboard.Key.esc:

                    self._terminate()  # Stop listener

            self._set_cmds()

    def start(self):

        with keyboard.Listener(on_press=self._on_press, 
                               on_release=self._on_release) as listener:

            listener.join()

if __name__ == "__main__":  

    keyb_cmds = RhcRefsFromKeyboard(namespace="kyon0", 
                            verbose=True)

    keyb_cmds.start()

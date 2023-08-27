from pynput import keyboard

from perf_sleep.pyperfsleep import PerfSleep

from control_cluster_utils.utilities.rhc_defs import RhcTaskRefs
from control_cluster_utils.utilities.defs import Journal

class RhcRefsFromKeyboard:

    def __init__(self):

        self.rhc_task_refs = []

        self._terminated = False

    def __del__(self):

        if not self._terminated:

            self._terminate()
    
    def _terminate(self):

        for i in range(0, self.cluster_size):

            self.rhc_task_refs[i].terminate()
        
        self._terminated = True
        
    def _on_press(self, key):

        try:

            print('Key {0} pressed.'.format(key.char))

        except AttributeError:

            print('Special key {0} pressed.'.format(key))

    def _on_release(self, key):

        print('Key {0} released.'.format(key))

        if key == keyboard.Key.esc:

            return False  # Stop listener
        
    def start(self):

        with keyboard.Listener(on_press=self._on_press, 
                               on_release=self._on_release) as listener:

            listener.join()

if __name__ == "__main__":  

    a = 1

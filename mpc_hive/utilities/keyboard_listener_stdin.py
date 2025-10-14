#!/usr/bin/env python

import threading
import sys
from select import select
import termios
import tty
import time

from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import Journal, LogType
from EigenIPC.PyEigenIPC import dtype

class KeyListenerStdin:
    def __init__(self, on_press=None, on_release=None, release_timeout=1.0):
        """
        Initialize the KeyListener.

        :param on_press: Callable to handle key press events.
        :param on_release: Callable to handle key release events.
        :param release_timeout: Timeout in seconds to detect key release.
        """
        self.on_press = on_press
        self.on_release = on_release
        self.done = False
        self.release_timeout = release_timeout
        self.pressed = {}  # Dictionary to keep track of pressed keys
        self.start_time = None
        self.release_thread = None
        self.lock = threading.Lock()  # Lock to protect pressed dictionary

    def __enter__(self):
        self._start_listener_thread()
        self._start_release_detection_thread()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _start_listener_thread(self):
        self.listener_thread = threading.Thread(target=self._listen_keys)
        self.listener_thread.daemon = True
        self.listener_thread.start()

    def _start_release_detection_thread(self):
        self.release_thread = threading.Thread(target=self._detect_key_release)
        self.release_thread.daemon = True
        self.release_thread.start()

    def stop(self):
        """Stop the listener thread."""
        if not self.done:
            self.done = True
            if self.listener_thread:
                if self.listener_thread.is_alive():
                    self.listener_thread.join()
            if self.release_thread:
                if self.release_thread.is_alive():
                    self.release_thread.join()

    def _listen_keys(self):
        """Thread to listen for key events."""
        settings = save_terminal_settings()
        while not self.done:                
            key = getKey(settings, timeout=None)  # blocking with timeout=None
            if key:
                self._handle_key_press(key)
                self._check_for_key_release()
       
        restore_terminal_settings(settings)

    def _handle_key_press(self, key):
        """Handle key press event."""
        with self.lock:  # Acquire the lock before modifying the pressed dictionary
            if key not in self.pressed:
                self.pressed[key] = time.time()  # Record time when key is pressed
            if self.on_press:
                self.on_press(key)

        # Exit the program if 'X' is pressed
        if key == 'X':
            Journal.log(self.__class__.__name__,
                "_handle_key_press",
                "X press detected -> exiting...",
                LogType.INFO,
                throw_when_excep = True)
            self.stop()

    def _check_for_key_release(self):
        """Check for key release using timeout."""
        current_time = time.time()
        with self.lock:  # Acquire the lock before reading from the pressed dictionary
            for key, press_time in list(self.pressed.items()):
                if current_time - press_time >= self.release_timeout:  # Check if timeout has passed
                    if self.on_release:
                        self.on_release(key)
                    del self.pressed[key]  # Remove the key from the pressed dictionary

    def _detect_key_release(self):
        """Background thread to periodically check for key release."""
        while not self.done:
            time.sleep(0.1)  # Check release status every 100ms
            self._check_for_key_release()


def getKey(settings, timeout):
    """Read a single keypress from stdin."""
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select([sys.stdin], [], [], timeout)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def save_terminal_settings():
    """Save current terminal settings."""
    return termios.tcgetattr(sys.stdin)


def restore_terminal_settings(old_settings):
    """Restore saved terminal settings."""
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    def on_key_press(key):
        print(f"Key pressed: {key}")

    def on_key_release(key):
        print(f"Key released: {key}")

    # Timeout for detecting key release is 1 second, polling rate 100Hz
    with KeyListenerStdin(on_press=on_key_press, 
        on_release=on_key_release, 
        release_timeout=0.1) as listener:
        try:
            while True:
                time.sleep(0.1)  # Keep the main thread alive
        except KeyboardInterrupt:
            print("Exiting...")





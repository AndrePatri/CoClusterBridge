#!/usr/bin/env python3
import threading
import sys
import os
from select import select
import termios
import tty
import time
from EigenIPC.PyEigenIPC import Journal, LogType

class KeyListenerStdin:
    def __init__(self, on_press=None, on_release=None, release_timeout=1.0, poll_interval=0.01):
        """
        on_press: callable(key) called for every key token read
        on_release: callable(key) called once when no new press for `key` seen for release_timeout
        release_timeout: seconds to treat 'no further press' as a release
        poll_interval: how often select() times out if no input (controls responsiveness)
        """
        self.on_press = on_press
        self.on_release = on_release
        self.release_timeout = release_timeout
        self.poll_interval = poll_interval

        self.done = False
        # pressed maps key -> {'last_seen': float, 'count': int}
        self.pressed = {}
        self.lock = threading.Lock()

        self.listener_thread = None
        self.release_thread = None

    def __enter__(self):
        self._start_listener_thread()
        self._start_release_detection_thread()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _start_listener_thread(self):
        self.listener_thread = threading.Thread(target=self._listen_keys, name="KeyListener")
        self.listener_thread.daemon = True
        self.listener_thread.start()

    def _start_release_detection_thread(self):
        self.release_thread = threading.Thread(target=self._detect_key_release, name="KeyReleaseDetector")
        self.release_thread.daemon = True
        self.release_thread.start()

    def stop(self):
        if not self.done:
            self.done = True
            # threads are daemonic; join briefly
            if self.listener_thread and self.listener_thread.is_alive():
                self.listener_thread.join(timeout=0.5)
            if self.release_thread and self.release_thread.is_alive():
                self.release_thread.join(timeout=0.5)

    def _listen_keys(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not self.done:
                rlist, _, _ = select([fd], [], [], self.poll_interval)
                if rlist:
                    try:
                        data = os.read(fd, 4096)
                    except OSError:
                        continue
                    if not data:
                        continue
                    tokens = self._parse_input_bytes(data)
                    for token in tokens:
                        self._handle_key_press(token)
                # loop continues, release thread handles timeouts
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _parse_input_bytes(self, b):
        """
        Convert raw bytes from terminal into a list of key tokens.
        - UTF-8 decode tolerant
        - Groups common escape sequences (like arrow keys) starting with ESC '[' ... letter
        """
        try:
            s = b.decode('utf-8', errors='replace')
        except Exception:
            s = ''.join(chr(x) for x in b)

        tokens = []
        i = 0
        L = len(s)
        while i < L:
            ch = s[i]
            if ch == '\x1b':  # escape sequence — try to grab CSI sequences like ESC [ ... <letter>
                if i + 1 < L and s[i+1] == '[':
                    j = i + 2
                    # read until a letter (A-Z or a-z) ends the sequence
                    while j < L and not s[j].isalpha():
                        j += 1
                    if j < L:
                        tokens.append(s[i:j+1])
                        i = j + 1
                        continue
                    else:
                        # partial; just append what we have
                        tokens.append(s[i:])
                        break
                else:
                    # single ESC
                    tokens.append(ch)
                    i += 1
            else:
                tokens.append(ch)
                i += 1
        return tokens

    def _handle_key_press(self, key):
        now = time.time()
        with self.lock:
            entry = self.pressed.get(key)
            if entry is None:
                self.pressed[key] = {'last_seen': now, 'count': 1}
            else:
                # update last_seen and bump count — this ensures repeated quick presses are tracked
                entry['last_seen'] = now
                entry['count'] += 1

        # always call on_press for each occurrence
        if self.on_press:
            try:
                self.on_press(key)
            except Exception as e:
                Journal.log(self.__class__.__name__, "_handle_key_press",
                            f"on_press handler raised: {e}", LogType.ERROR, throw_when_excep=False)

        # convenience: immediate exit on 'X' (same as original)
        if key == 'X':
            Journal.log(self.__class__.__name__,
                        "_handle_key_press",
                        "X press detected -> exiting...",
                        LogType.INFO,
                        throw_when_excep=True)
            self.stop()

    def _check_for_key_release_once(self):
        """Check pressed keys and emit release where last_seen older than timeout."""
        now = time.time()
        to_release = []
        with self.lock:
            for key, entry in list(self.pressed.items()):
                if now - entry['last_seen'] >= self.release_timeout:
                    to_release.append((key, entry.get('count', 1)))
                    del self.pressed[key]

        # call on_release outside the lock
        for key, count in to_release:
            if self.on_release:
                try:
                    # call with just key to preserve compatibility
                    self.on_release(key)
                except TypeError:
                    # If user supplied a handler that accepts (key, count), support it too
                    try:
                        self.on_release(key, count)
                    except Exception as e:
                        Journal.log(self.__class__.__name__, "_check_for_key_release_once",
                                    f"on_release handler raised: {e}", LogType.ERROR, throw_when_excep=False)
                except Exception as e:
                    Journal.log(self.__class__.__name__, "_check_for_key_release_once",
                                f"on_release handler raised: {e}", LogType.ERROR, throw_when_excep=False)

    def _detect_key_release(self):
        while not self.done:
            time.sleep(min(0.1, self.release_timeout / 4.0))
            self._check_for_key_release_once()


if __name__ == "__main__":
    def on_key_press(key):
        print(f"Key pressed: {repr(key)}")

    def on_key_release(key):
        print(f"Key released: {repr(key)}")

    with KeyListenerStdin(on_press=on_key_press, on_release=on_key_release, release_timeout=0.1) as listener:
        try:
            while not listener.done:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Exiting...")
            listener.stop()

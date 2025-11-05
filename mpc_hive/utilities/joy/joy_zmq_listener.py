#!/usr/bin/env python3
# joy_sub.py
# Simple ZeroMQ subscriber that connects to the publisher and prints/decodes joystick state.
#
# Usage:
#   python joy_sub.py
#   python joy_sub.py --connect 192.168.1.10:5556 --topic joy

import argparse
import zmq
import json
import time
import threading
import numpy as np
from typing import Optional, Callable


class JoyListenerZMQ:
    """
    ZeroMQ-based joystick state listener.

    - Connects to a PUB socket and subscribes to `topic`.
    - Runs a background thread to receive (non-busy) and parse messages.
    - Keeps the latest joystick state in numpy arrays:
       - sticks: shape (4,) floats -> [left_x, left_y, right_x, right_y]
       - triggers: shape (2,) floats -> [left_trigger, right_trigger]
       - bumpers: shape (2,) bools -> [left_bumper, right_bumper]
       - face: shape (4,) bools -> [X, B, A, Y]   (keeps this naming to match your previous "xbay")
       - back_start_home: shape (3,) bools -> [back, start, home]
       - hat: shape (2,) ints -> (x, y) e.g. (-1,0,1)
    - Optional callback on_message(payload) is called for each received payload.
    """

    def __init__(
        self,
        connect: str = "localhost:5556",
        topic: str = "joy",
        poll_interval: float = 0.01,
        on_message: Optional[Callable[[dict], None]] = None,
        debug: bool = False
    ):
        self.debug=debug

        self.connect = connect
        self.topic = topic
        self.poll_interval = float(poll_interval)
        self.on_message = on_message

        # thread control
        self.done = False
        self.listener_thread: Optional[threading.Thread] = None

        # ZMQ setup
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        self.connect_addr = f"tcp://{self.connect}"
        print("[JoyListenerZMQ]: Connecting to", self.connect_addr)
        self.sock.connect(self.connect_addr)

        # subscribe to topic
        self.topic_bytes = self.topic.encode("utf-8")
        self.sock.setsockopt(zmq.SUBSCRIBE, self.topic_bytes)

        # state holders (default neutral)
        self.sticks = np.zeros(4, dtype=np.float32)  # left_x,left_y,right_x,right_y
        self.triggers = np.zeros(2, dtype=np.float32)  # left, right
        self.bumpers = np.zeros(2, dtype=bool)  # left, right
        self.face = np.zeros(4, dtype=bool)  # X, B, A, Y (keeps the size)
        self.back_start_home = np.zeros(3, dtype=bool)  # back, start, home
        self.hat = np.array([0, 0], dtype=int)  # (x, y) values from hat; -1/0/1

        # raw last payload storage
        self.seq = None
        self.ts = None
        self.name = "<unknown>"
        self.axes = []
        self.buttons = []
        self.hats = []

    def __enter__(self):
        self._start_listener()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _start_listener(self):
        if self.listener_thread and self.listener_thread.is_alive():
            return
        self.listener_thread = threading.Thread(target=self._poll_joy, name="JoyListenerZMQ")
        self.listener_thread.daemon = True
        self.listener_thread.start()

    def _poll_joy(self):
        """
        Poll ZMQ socket with a Poller (no busy loop). When a message arrives,
        decode JSON and store the data. Calls on_message(payload) if provided.
        """
        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)

        try:
            while not self.done:
                # poller timeout in milliseconds
                events = dict(poller.poll(int(self.poll_interval * 1000)))
                if self.sock in events:
                    try:
                        frames = self.sock.recv_multipart(flags=0)
                    except zmq.ZMQError as e:
                        # interrupted or socket closed
                        if self.done:
                            break
                        print("[JoyListenerZMQ] ZMQ recv error:", e)
                        continue

                    if len(frames) != 2:
                        print("Unexpected frame count:", len(frames))
                        continue

                    topic_frame, jbytes = frames
                    # check topic to be safe
                    if topic_frame != self.topic_bytes:
                        # ignore unexpected topic
                        continue

                    try:
                        payload = json.loads(jbytes.decode("utf-8"))
                    except Exception as e:
                        print("JSON decode error:", e)
                        continue

                    # store internal arrays
                    self._store_payload_data(payload)

                    # call optional callback
                    if self.on_message:
                        try:
                            self.on_message(payload)
                        except Exception as e:
                            print("on_message callback error:", e)
                # else: no event, just loop again (non-busy thanks to poll timeout)
        except KeyboardInterrupt:
            print("Subscriber stopped by user")
        finally:
            try:
                self.sock.close(linger=0)
            except Exception:
                pass
            try:
                self.ctx.term()
            except Exception:
                pass

    def _store_payload_data(self, payload: dict):
        """
        Extracts seq, timestamp, axes, buttons, hats and writes them into numpy arrays
        with safe bounds checks. Uses common Xbox-style indexes but is robust to
        shorter arrays (it will only fill what exists).
        """
        self.seq = payload.get("seq")
        self.ts = payload.get("timestamp", time.time())
        state = payload.get("state", {})
        self.name = state.get("name", "<unknown>")

        raw_axes = state.get("axes", [])
        raw_buttons = state.get("buttons", [])
        raw_hats = state.get("hats", [])

        # store raw copies for debugging / advanced uses
        self.axes = raw_axes
        self.buttons = raw_buttons
        self.hats = raw_hats

        # Clear previous arrays to defaults
        self.sticks[:] = 0.0
        self.triggers[:] = 0.0
        self.bumpers[:] = False
        self.face[:] = False
        self.back_start_home[:] = False
        self.hat[:] = 0

        # Typical mapping (may vary by platform):
        # axes: 0:left_x, 1:left_y, 2:left_trigger or sometimes RT, 3:right_x, 4:right_y, 5:right_trigger
        # We'll map what exists conservatively.
        def get_axis(i, default=0.0):
            return float(raw_axes[i]) if (i < len(raw_axes)) else default

        # left stick
        self.sticks[0] = get_axis(0)
        self.sticks[1] = - get_axis(1)
        # right stick
        # Some drivers put right_x at index 2; many put it at 3. We'll attempt common indices.
        # Try index 3 then 2 for right_x; try 4 then 3 for right_y.
        if len(raw_axes) > 3:
            self.sticks[2] = get_axis(3)
            self.sticks[3] = - get_axis(4) if len(raw_axes) > 4 else 0.0
        elif len(raw_axes) > 2:
            # fallback: assume [left_x, left_y, right_x, right_y...] or triggers
            # try to place right_x at 2 and right_y at 3 if available
            self.sticks[2] = get_axis(2)
            self.sticks[3] = -get_axis(3) if len(raw_axes) > 3 else 0.0

        # triggers: often exposed as axes (range -1..1 or 0..1). Many mappings:
        # If axes length >= 6, assume axes[2] is left trigger and axes[5] right trigger
        if len(raw_axes) >= 6:
            self.triggers[0] = get_axis(2)
            self.triggers[1] = get_axis(5)
        elif len(raw_axes) == 3:
            # sometimes triggers are combined; fallback: keep 0
            self.triggers[0] = 0.0
            self.triggers[1] = 0.0
        elif len(raw_axes) == 4:
            # maybe triggers absent; leave zeros
            pass
        elif len(raw_axes) >= 5:
            # try axes[2] and axes[4]
            self.triggers[0] = get_axis(2)
            self.triggers[1] = get_axis(4)

        # Buttons: typical mapping (but may differ):
        # 0:A, 1:B, 2:X, 3:Y, 4:LB, 5:RB, 6:BACK, 7:START, 8:GUIDE, 9:L3, 10:R3
        def get_button(i, default=False):
            return bool(raw_buttons[i]) if (i < len(raw_buttons)) else default

        # face buttons (A,B,X,Y) -> we'll map to [X, B, A, Y] only because you used that earlier
        # But typical order is [A,B,X,Y] -> indices 0..3. We'll map to face as [X,B,A,Y] to match your previous variable names:
        # To avoid confusion, we will store as: face = [X, B, A, Y] if all available,
        # otherwise fill in what exists using the canonical A,B,X,Y mapping.
        a = get_button(0)
        b = get_button(1)
        x = get_button(2)
        y = get_button(3)
        # store in the "face" array as [X, B, A, Y] per your previous variable naming (`xbay`)
        self.face[0] = x
        self.face[1] = b
        self.face[2] = a
        self.face[3] = y

        # bumpers
        self.bumpers[0] = get_button(4)  # LB
        self.bumpers[1] = get_button(5)  # RB

        # back/start/home (guide)
        self.back_start_home[0] = get_button(6)  # back
        self.back_start_home[1] = get_button(7)  # start
        # guide/home may be button 8 (or absent)
        self.back_start_home[2] = get_button(8)

        # hats: take first hat if present
        if len(raw_hats) >= 1:
            hat0 = raw_hats[0]
            try:
                hx = int(hat0[0])
                hy = int(hat0[1])
                self.hat[0] = hx
                self.hat[1] = hy
            except Exception:
                # fallback
                self.hat[:] = 0

        # compose debug/info string
        self.info_str = (
            f"[{time.strftime('%H:%M:%S', time.localtime(self.ts))}] "
            f"seq={self.seq} device='{self.name}' axes={len(self.axes)} buttons={len(self.buttons)} hats={len(self.hats)}"
        )

    def stop(self):
        """
        Request the listener to stop and join the thread briefly.
        """
        if not self.done:
            self.done = True
            # closing socket will interrupt poll/recv
            try:
                self.sock.close(linger=0)
            except Exception:
                pass
            try:
                self.ctx.term()
            except Exception:
                pass

            # join thread
            if self.listener_thread and self.listener_thread.is_alive():
                self.listener_thread.join(timeout=1.0)

    # def pretty_print_payload(self, payload: dict):
    #     """
    #     Pretty print a payload (same as earlier helper).
    #     """
    #     seq = payload.get("seq")
    #     ts = payload.get("timestamp")
    #     state = payload.get("state", {})
    #     name = state.get("name", "<unknown>")
    #     axes = state.get("axes", [])
    #     buttons = state.get("buttons", [])
    #     hats = state.get("hats", [])
    #     print(
    #         f"[{time.strftime('%H:%M:%S', time.localtime(ts))}] seq={seq} device='{name}' axes={len(axes)} buttons={len(buttons)} hats={len(hats)}"
    #     )
    #     # Small summary of first few values for readability:
    #     print("  axes:", [round(a, 3) for a in axes[:8]])
    #     print("  buttons:", buttons[:16])
    #     print("  hats:", hats)
    #     print("-" * 50)

    def pretty_print_payload(self, payload: dict = None):
        """
        Print the latest state from the listener's numpy arrays (thread-safe-ish).
        If payload is provided and arrays are missing, falls back to printing payload.
        """
        try:
            # Info/header (use stored info_str if available)
            header = getattr(self, "info_str", None)
            if header:
                print(header)
            else:
                if payload:
                    ts = payload.get("timestamp")
                    seq = payload.get("seq")
                    name = payload.get("state", {}).get("name", "<unknown>")
                    print(f"[{time.strftime('%H:%M:%S', time.localtime(ts))}] seq={seq} device='{name}'")

            # Copy arrays so we don't print while they are being updated
            sticks = self.sticks.copy()       # left_x, left_y, right_x, right_y
            triggers = self.triggers.copy()   # left, right
            bumpers = self.bumpers.copy()     # left, right
            face = self.face.copy()           # X, B, A, Y (kept as in class)
            back = self.back_start_home.copy()# back, start, home
            hat = self.hat.copy()             # (x, y)
            raw_axes = list(self.axes) if hasattr(self, "axes") else []
            raw_buttons = list(self.buttons) if hasattr(self, "buttons") else []
            raw_hats = list(self.hats) if hasattr(self, "hats") else []

            # Format and print
            # Round floats for readability
            def r(x): 
                try:
                    return round(float(x), 3)
                except Exception:
                    return x

            print("  sticks (left_x,left_y,right_x,right_y):", [r(v) for v in sticks])
            print("  triggers (L,R):", [r(v) for v in triggers])
            print("  bumpers (LB,RB):", [bool(x) for x in bumpers])
            print("  face (X,B,A,Y):", [bool(x) for x in face])
            print("  back/start/home:", [bool(x) for x in back])
            print("  hat (x,y):", (int(hat[0]), int(hat[1])))

            # Also show raw lists for debugging (truncated)
            if self.debug:
                print("  raw axes (first 8):", [r(a) for a in raw_axes[:8]])
                print("  raw buttons (first 16):", raw_buttons[:16])
                print("  raw hats:", raw_hats)
                print("-" * 50)

        except Exception as e:
            # Don't crash the whole program if printing fails
            print("pretty_print_payload error:", e)
            # fallback: if payload present, print minimal info from it
            if payload:
                seq = payload.get("seq")
                ts = payload.get("timestamp")
                state = payload.get("state", {})
                name = state.get("name", "<unknown>")
                axes = state.get("axes", [])
                buttons = state.get("buttons", [])
                hats = state.get("hats", [])
                print(f"[{time.strftime('%H:%M:%S', time.localtime(ts))}] seq={seq} device='{name}' axes={len(axes)} buttons={len(buttons)} hats={len(hats)}")

def main():
    parser = argparse.ArgumentParser(description="ZeroMQ joystick subscriber (threaded, non-busy).")
    parser.add_argument("--connect", default="localhost:5556", help="Publisher address to connect to (host:port). Default localhost:5556")
    parser.add_argument("--topic", default="joy", help="Topic to subscribe to (default 'joy')")
    parser.add_argument("--poll-interval", type=float, default=0.01, help="Poll interval seconds (default 0.01)")
    args = parser.parse_args()

    # A simple on_message callback that prints the payload summary
    def on_message(payload):
        # Print the payload via the object's pretty printer — but we need access to listener.
        # We'll capture listener from outer scope by setting it after creation. Use fallback print.
        if hasattr(listener, "pretty_print_payload"):
            listener.pretty_print_payload(payload)
        else:
            # fallback minimal print
            seq = payload.get("seq")
            ts = payload.get("timestamp")
            print(f"[{time.strftime('%H:%M:%S', time.localtime(ts))}] seq={seq}")

    # create listener and run until Ctrl-C
    listener = JoyListenerZMQ(connect=args.connect, topic=args.topic, poll_interval=args.poll_interval, on_message=on_message)

    # start listening using context manager (optional)
    listener._start_listener()
    print("Listener started. Press Ctrl-C to exit.")
    try:
        while not listener.done:
            # main loop can do other work; here we sleep to be idle but responsive
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        listener.stop()


if __name__ == "__main__":
    main()

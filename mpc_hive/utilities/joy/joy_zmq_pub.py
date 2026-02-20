#!/usr/bin/env python3
# joy_pub.py
# Very simple ZeroMQ publisher that reads a joystick (Xbox-style) via pygame
# and publishes the full device state repeatedly.
#
# Usage:
#   python joy_pub.py            # connects to tcp://localhost:5556
#   python joy_pub.py --connect 192.168.1.10:5557
#   python joy_pub.py --rate 60  # publish at 60 Hz
#   python joy_pub.py --input-mode jsdev --jsdev /dev/input/js0

import os
# Use dummy video driver so pygame can init joysticks on headless Linux.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
# Avoid ALSA device requirements inside containers.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import time
import json
import argparse
import errno
import struct
import array
import fcntl
from pathlib import Path
import zmq

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80


def read_state_pygame(js, pygame_mod):
    """Return a dict with full joystick state: axes, buttons, hats"""
    # pump pygame events so axis/button states update
    try:
        pygame_mod.event.pump()
    except Exception:
        # Some headless/container SDL setups do not expose an event/video subsystem.
        # Keep publishing using direct joystick polling.
        pass
    axes = [js.get_axis(i) for i in range(js.get_numaxes())]
    buttons = [js.get_button(i) for i in range(js.get_numbuttons())]
    hats = [js.get_hat(i) for i in range(js.get_numhats())]
    return {"name": js.get_name(),
            "num_axes": js.get_numaxes(),
            "num_buttons": js.get_numbuttons(),
            "num_hats": js.get_numhats(),
            "axes": axes,
            "buttons": buttons,
            "hats": hats}


def _open_jsdev(path: str):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)

    # ioctls from linux/joystick.h
    jsio_cgaxes = 0x80016A11
    jsio_cgbuttons = 0x80016A12
    jsio_cgname = 0x80006A13

    n_axes = array.array("B", [0])
    n_buttons = array.array("B", [0])
    fcntl.ioctl(fd, jsio_cgaxes, n_axes, True)
    fcntl.ioctl(fd, jsio_cgbuttons, n_buttons, True)

    name = path
    name_buf = array.array("b", [0] * 128)
    try:
        fcntl.ioctl(fd, jsio_cgname + (len(name_buf) << 16), name_buf, True)
        raw = name_buf.tobytes().split(b"\x00", 1)[0]
        if raw:
            name = raw.decode("utf-8", errors="ignore")
    except OSError:
        pass

    axes = [0.0] * int(n_axes[0])
    buttons = [0] * int(n_buttons[0])
    return fd, name, axes, buttons


def _drain_jsdev_events(fd: int, axes, buttons):
    while True:
        try:
            ev = os.read(fd, 8)
        except BlockingIOError:
            break
        except OSError as e:
            if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                break
            raise

        if len(ev) != 8:
            break
        _, value, ev_type, number = struct.unpack("IhBB", ev)
        ev_type &= ~JS_EVENT_INIT

        if ev_type == JS_EVENT_AXIS and number < len(axes):
            axes[number] = max(-1.0, min(1.0, float(value) / 32767.0))
        elif ev_type == JS_EVENT_BUTTON and number < len(buttons):
            buttons[number] = 1 if value else 0


def read_state_jsdev(fd: int, name: str, axes, buttons):
    _drain_jsdev_events(fd, axes, buttons)
    return {
        "name": name,
        "num_axes": len(axes),
        "num_buttons": len(buttons),
        "num_hats": 0,
        "axes": list(axes),
        "buttons": list(buttons),
        "hats": [],
    }


def main():
    parser = argparse.ArgumentParser(description="Joystick -> ZeroMQ publisher")
    parser.add_argument("--connect", default="localhost:5556",
                        help="Listener address (host:port) to connect to. Default 'localhost:5556'")
    parser.add_argument("--topic", default="joy", help="Topic string (default 'joy')")
    parser.add_argument("--rate", type=float, default=30.0, help="Publish rate Hz (default 30)")
    parser.add_argument("--delay-after-connect", type=float, default=0.2,
                        help="Short delay after connect to allow subscriber bind/subscription setup")
    parser.add_argument("--input-mode", choices=["pygame", "jsdev"], default="pygame",
                        help="Input backend: 'pygame' (SDL) or 'jsdev' (/dev/input/jsX).")
    parser.add_argument("--pygame-index", type=int, default=-1,
                        help="Pygame joystick index. Default -1 selects the most joystick-like device.")
    parser.add_argument("--jsdev", default="/dev/input/js0",
                        help="Joystick device path when --input-mode jsdev (default /dev/input/js0).")
    args = parser.parse_args()

    zmq_ctx = zmq.Context()
    sock = zmq_ctx.socket(zmq.PUB)
    connect_addr = f"tcp://{args.connect}"
    print("Connecting publisher to", connect_addr)
    sock.connect(connect_addr)

    # short delay so subscribers that connect immediately have time to connect
    time.sleep(args.delay_after_connect)

    pygame_mod = None
    js = None
    js_fd = None
    state_reader = None

    if args.input_mode == "pygame":
        import pygame as pygame_mod
        # initialize pygame subsystems in a container-safe way
        pygame_mod.init()
        pygame_mod.joystick.init()
        js_count = pygame_mod.joystick.get_count()
        if js_count == 0:
            js_nodes = sorted(str(p) for p in Path("/dev/input").glob("js*"))
            evt_nodes = sorted(str(p) for p in Path("/dev/input").glob("event*"))[:8]
            print("No joystick found by SDL. /dev/input js nodes:", js_nodes)
            print("No joystick found by SDL. /dev/input event sample:", evt_nodes)
            print("No joystick found. Exiting.")
            return

        if args.pygame_index >= 0:
            chosen_idx = min(args.pygame_index, js_count - 1)
        else:
            # Auto-pick the most joystick-like device (prefer more axes/buttons).
            chosen_idx = 0
            best_score = -1
            for i in range(js_count):
                cand = pygame_mod.joystick.Joystick(i)
                cand.init()
                score = cand.get_numaxes() * 100 + cand.get_numbuttons()
                cand.quit()
                if score > best_score:
                    best_score = score
                    chosen_idx = i

        js = pygame_mod.joystick.Joystick(chosen_idx)
        js.init()
        print(f"Using joystick {chosen_idx} (pygame):", js.get_name())
        state_reader = lambda: read_state_pygame(js, pygame_mod)
    else:
        js_fd, js_name, axes, buttons = _open_jsdev(args.jsdev)
        print(f"Using joystick (jsdev): {js_name} at {args.jsdev} with "
              f"{len(axes)} axes and {len(buttons)} buttons")
        state_reader = lambda: read_state_jsdev(js_fd, js_name, axes, buttons)

    seq = 0
    interval = 1.0 / max(0.001, args.rate)

    try:
        while True:
            state = state_reader()
            payload = {
                "seq": seq,
                "timestamp": time.time(),
                "state": state
            }
            jbytes = json.dumps(payload, allow_nan=False).encode("utf-8")
            # send multipart: topic frame + json frame
            sock.send_multipart([args.topic.encode("utf-8"), jbytes])
            # for debug print a small summary to console (comment out if noisy)
            print(f"SENT seq={seq} axes={len(state['axes'])} btns={len(state['buttons'])} hats={len(state['hats'])}")
            seq += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        if js is not None:
            js.quit()
        if pygame_mod is not None:
            pygame_mod.quit()
        if js_fd is not None:
            os.close(js_fd)
        sock.close()
        zmq_ctx.term()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# joy_pub.py
# Very simple ZeroMQ publisher that reads a joystick (Xbox-style) via pygame
# and publishes the full device state repeatedly.
#
# Usage:
#   python joy_pub.py            # binds to tcp://*:5556
#   python joy_pub.py --bind 0.0.0.0:5557
#   python joy_pub.py --rate 60  # publish at 60 Hz

import os
# Use dummy video driver so pygame can init joysticks on headless Linux.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import time
import json
import argparse
import zmq
import pygame

def read_state(js):
    """Return a dict with full joystick state: axes, buttons, hats"""
    # pump pygame events so axis/button states update
    pygame.event.pump()
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

def main():
    parser = argparse.ArgumentParser(description="Joystick -> ZeroMQ publisher")
    parser.add_argument("--bind", default="*:5556",
                        help="Bind address (host:port) for the publisher. Default '*:5556'")
    parser.add_argument("--topic", default="joy", help="Topic string (default 'joy')")
    parser.add_argument("--rate", type=float, default=30.0, help="Publish rate Hz (default 30)")
    parser.add_argument("--delay-after-bind", type=float, default=0.2,
                        help="Short delay after bind to allow subscribers to connect")
    args = parser.parse_args()

    zmq_ctx = zmq.Context()
    sock = zmq_ctx.socket(zmq.PUB)
    bind_addr = f"tcp://{args.bind}"
    print("Binding publisher to", bind_addr)
    sock.bind(bind_addr)

    # short delay so subscribers that connect immediately have time to connect
    time.sleep(args.delay_after_bind)

    # initialize pygame joystick
    pygame.init()
    pygame.joystick.init()
    js_count = pygame.joystick.get_count()
    if js_count == 0:
        print("No joystick found. Exiting.")
        return

    js = pygame.joystick.Joystick(0)
    js.init()
    print("Using joystick 0:", js.get_name())
    seq = 0
    interval = 1.0 / max(0.001, args.rate)

    try:
        while True:
            state = read_state(js)
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
        js.quit()
        pygame.quit()
        sock.close()
        zmq_ctx.term()

if __name__ == "__main__":
    main()
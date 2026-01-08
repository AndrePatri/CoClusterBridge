# RefsFromJoy - joystick-driven class that replicates RefsFromKeyboard write behavior
# Maps an Xbox-style controller (as delivered by JoyListenerZMQ) to the same
# shared memory writes that RefsFromKeyboard used (contacts, phase id, base height,
# flight params, navigation/twist, etc.).
#
# Reasonable mapping (documented below) — adjust to taste.

from aug_mpc.utils.shared_data.agent_refs import AgentRefs
from mpc_hive.utilities.shared_data.rhc_data import RobotState
from mpc_hive.utilities.math_utils import world2base_frame_twist

from EigenIPC.PyEigenIPCExt.wrappers.shared_data_view import SharedTWrapper
from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import Journal, LogType
from EigenIPC.PyEigenIPC import dtype

import math
import time
import numpy as np

# Import the provided JoyListenerZMQ (assumes it's importable from your path)
# If it's in another module, change the import accordingly.
from mpc_hive.utilities.joy.joy_zmq_listener import JoyListenerZMQ

from typing import Optional, Callable, Any

class RefsFromJoy:

    def __init__(self,
                 namespace: str,
                 shared_refs,
                 verbose: bool = False,
                 agent_refs_world: bool = False,
                 env_idx: int = None,
                 hold_time: float = 0.15):
        self.namespace = namespace
        self._verbose = verbose
        self._agent_refs_world = agent_refs_world
        self._env_idx = env_idx
        self.hold_time = float(hold_time)
        self._closed = False

        # optional old shared_refs (for contact_flags, phase_id, flight_settings, etc.)
        self._shared_refs = shared_refs

        # navigation / twist flags (same as RefsFromJoy)
        self.enable_linvel = False
        self.enable_omega = False
        self.enable_pos = False
        self.enable_linvelz = False

        self.dpos = 0.1
        self.dxy = 0.05
        self._dtwist = 1.0 * math.pi / 180.0

        self._v_magnitude = 0.0
        self._heading = 0.0
        
        self._max_vxy_magn = 1.5
        self._max_vz_magn = 1.0
        self._max_pitch_rate = 0.8
        self._max_roll_rate = 0.8
        self._max_yaw_rate = 0.8

        # flight params state (kept similar to keyboard class)
        self._enable_flight_param_change = False
        self._d_flength_enabled = False
        self._d_fapex_enabled = False
        self._d_fend_enabled = False
        self._d_fparam_enabled_contact_i = [False]*4
        self._d_flight_length = 1
        self._d_flight_apex = 0.01
        self._d_flight_end = 0.01

        self.enable_heightchange = False
        self.height_dh = 0.02

        # phase id
        self.enable_phase_id_change = False
        self._phase_id_current = 0

        # contact mapping (if you used a custom mapping string in keyboard class)
        self._contact_mapping = [0,1,2,3]

        # cluster index placeholder
        self.cluster_idx = -1
        self.cluster_idx_np = np.array(self.cluster_idx)

        # agent_refs / robot_state
        self._robot_state = None

        # hold toggles same pattern as RefsFromJoy
        self._hold_since = {"omega": None, "linvel": None, "pos": None, "linvelz": None}
        self._hold_triggered = {"omega": False, "linvel": False, "pos": False, "linvelz": False}

        # previous joy snapshot for edge detection
        self._prev_face = np.zeros(4, dtype=bool)
        self._prev_bumpers = np.zeros(2, dtype=bool)
        self._prev_back_start_home = np.zeros(3, dtype=bool)
        self._prev_triggers = np.zeros(2, dtype=float)
        self._prev_hat = np.array([0,0], dtype=int)

        self._init_shared_data()

    def _init_shared_data(self):
        # env index wrapper if needed
        self.env_index = None
        if self._env_idx is None:
            self.env_index = SharedTWrapper(namespace=self.namespace,
                                           basename="EnvSelector",
                                           is_server=False,
                                           verbose=True,
                                           vlevel=VLevel.V2,
                                           safe=False,
                                           dtype=dtype.Int)
            self.env_index.run()

        self._shared_refs.run()
        
        self._robot_state = RobotState(namespace=self.namespace,
                                       is_server=False,
                                       safe=False,
                                       verbose=True,
                                       vlevel=VLevel.V2)
        self._robot_state.run()

        # convenience buffers
        self._current_twist_ref_world = np.full_like(
            self._shared_refs.rob_refs.root_state.get(data_type="twist", robot_idxs=np.array(self.cluster_idx)),
            fill_value=0.0).reshape(-1)
        self._current_twist_ref_base = np.full_like(self._current_twist_ref_world, fill_value=0.0).reshape(1, -1)
        self._current_pos_ref = np.full_like(
            self._shared_refs.rob_refs.root_state.get(data_type="p", robot_idxs=np.array(self.cluster_idx)),
            fill_value=0.0).reshape(-1)

    def __del__(self):
        if not self._closed:
            self._close()

    def _close(self):
        if self._shared_refs is not None:
            self._shared_refs.close()
        if self._robot_state is not None:
            self._robot_state.close()
        if self.env_index is not None:
            self.env_index.close()
        if self._shared_refs is not None:
            try:
                self._shared_refs.close()
            except Exception:
                pass
        self._closed = True

    # -------------------
    # Low level writers (same as keyboard class)
    # -------------------
    def _update_base_height(self, decrement = False):
        if self._shared_refs is None:
            return
        current_p_ref = self._shared_refs.rob_refs.root_state.get(data_type="p", robot_idxs=self.cluster_idx_np)
        if decrement:
            new_height_ref = current_p_ref[2] - self.height_dh
        else:
            new_height_ref = current_p_ref[2] + self.height_dh
        current_p_ref[2] = new_height_ref
        self._shared_refs.rob_refs.root_state.set(data_type="p", data=current_p_ref,
                                                  robot_idxs=self.cluster_idx_np)

    def _set_contacts(self, contact_idx: int, is_contact: bool = True):
        if self._shared_refs is None:
            return
        contact_flags = self._shared_refs.contact_flags.get_numpy_mirror()
        mapped = self._contact_mapping[contact_idx]
        contact_flags[self.cluster_idx, mapped] = is_contact
        self._shared_refs.contact_flags.synch_retry(row_index=self.cluster_idx, col_index=0, 
                                                n_rows=1, n_cols=self._shared_refs.contact_flags.n_cols,
                                                read=False)

    def _update_phase_id(self, phase_id: int = -1):
        if self._shared_refs is None:
            return
        phase_id_shared = self._shared_refs.phase_id.get_numpy_mirror()
        phase_id_shared[self.cluster_idx, :] = phase_id
        self._shared_refs.phase_id.synch_retry(row_index=self.cluster_idx, col_index=0, 
                                                n_rows=1, n_cols=self._shared_refs.phase_id.n_cols,
                                                read=False)
        self._phase_id_current = phase_id

    def _update_flight_params(self, contact_idx: int, increment: bool = True):
        if self._shared_refs is None:
            return
        # follow same logic as your keyboard class: adjust len/apex/end if enabled
        if self._d_flength_enabled and self._d_fparam_enabled_contact_i[contact_idx]:
            length_now = self._shared_refs.flight_settings_req.get(data_type="len_remain",
                                                               robot_idxs=self.cluster_idx,
                                                               contact_idx=contact_idx)
            length_now = length_now + (self._d_flight_length if increment else -self._d_flight_length)
            self._shared_refs.flight_settings_req.set(data=np.array(length_now),
                                                 data_type="len_remain",
                                                 robot_idxs=self.cluster_idx,
                                                 contact_idx=contact_idx)
        if self._d_fapex_enabled and self._d_fparam_enabled_contact_i[contact_idx]:
            apex_now = self._shared_refs.flight_settings_req.get(data_type="apex_dpos",
                                                             robot_idxs=self.cluster_idx,
                                                             contact_idx=contact_idx)
            apex_now = apex_now + (self._d_flight_apex if increment else -self._d_flight_apex)
            self._shared_refs.flight_settings_req.set(data=np.array(apex_now),
                                                 data_type="apex_dpos",
                                                 robot_idxs=self.cluster_idx,
                                                 contact_idx=contact_idx)
        if self._d_fend_enabled and self._d_fparam_enabled_contact_i[contact_idx]:
            end_now = self._shared_refs.flight_settings_req.get(data_type="end_dpos",
                                                            robot_idxs=self.cluster_idx,
                                                            contact_idx=contact_idx)
            end_now = end_now + (self._d_flight_end if increment else -self._d_flight_end)
            self._shared_refs.flight_settings_req.set(data=np.array(end_now),
                                                 data_type="end_dpos",
                                                 robot_idxs=self.cluster_idx,
                                                 contact_idx=contact_idx)

    # -------------------
    # High-level joystick -> ref logic (Copied/adapted from RefsFromJoy)
    # -------------------
    def _check_and_toggle(self, name: str, pressed: bool):
        now = time.time()
        if name not in self._hold_since:
            return
        if pressed:
            if self._hold_since[name] is None:
                self._hold_since[name] = now
            else:
                duration = now - self._hold_since[name]
                if duration >= self.hold_time and not self._hold_triggered[name]:
                    if name == "omega":
                        self.enable_omega = not self.enable_omega
                        info = f"Omega change enabled: {self.enable_omega}"
                        Journal.log(self.__class__.__name__, "_set_omega", info, LogType.INFO, throw_when_excep=True)
                    elif name == "linvel":
                        self.enable_linvel = not self.enable_linvel
                        info = f"Linvel xy enabled: {self.enable_linvel}"
                        Journal.log(self.__class__.__name__, "_set_linvel", info, LogType.INFO, throw_when_excep=True)
                    elif name == "pos":
                        self.enable_pos = not self.enable_pos
                        info = f"pos reference change: {self.enable_pos}"
                        Journal.log(self.__class__.__name__, "_set_position", info, LogType.INFO, throw_when_excep=True)
                    elif name == "linvelz":
                        self.enable_linvelz = not self.enable_linvelz
                        info = f"linvel z enabled: {self.enable_linvelz}"
                        Journal.log(self.__class__.__name__, "_set_linvel", info, LogType.INFO, throw_when_excep=True)
                    self._hold_triggered[name] = True
        else:
            self._hold_since[name] = None
            self._hold_triggered[name] = False

    def _norm_trigger(self, val: float) -> float:
            # Heuristic normalization to [0,1]
            if val < -0.1:
                # assume in [-1,1] -> map rest=-1 -> 0, pressed=+1 -> 1
                return float(np.clip((val + 1.0) / 2.0, 0.0, 1.0))
            else:
                # assume already in [0,1]
                return float(np.clip(val, 0.0, 1.0))

    def _set_omega(self, joy):
        twist_ref = self._current_twist_ref_world
        if not self.enable_omega:
            twist_ref[3:] = 0.0
            return
        # twist_ref[3] = 0.0
        # twist_ref[4] = 0.0
        lsx = float(joy.sticks[0])
        lsy = float(joy.sticks[1])
        lt = float(joy.triggers[0])
        rt = float(joy.triggers[1])
        lt_n = self._norm_trigger(lt)
        rt_n = self._norm_trigger(rt)

        # Map left stick directly to roll/pitch rates (omega x/y)
        # Optionally add a small deadzone to avoid jitter
        deadzone = float(getattr(self, "dxy", 0.05))
        if abs(lsx) <= deadzone:
            roll_cmd = 0.0
        else:
            roll_cmd = np.clip(lsx, -1.0, 1.0) * float(self._max_roll_rate)
        
        if abs(lsy) <= deadzone:
            pitch_cmd = 0.0
        else:
            pitch_cmd = np.clip(lsy, -1.0, 1.0) * float(self._max_pitch_rate)

        # yaw = RT_positive minus LT_negative
        yaw_cmd = -(rt_n - lt_n) * float(self._max_yaw_rate)
        # small deadzone so tiny trigger jitter doesn't move yaw
        if abs(yaw_cmd) < 1e-4:
            yaw_cmd = 0.0

        twist_ref[3] = float(np.clip(roll_cmd, -self._max_roll_rate, self._max_roll_rate))
        twist_ref[4] = float(np.clip(pitch_cmd, -self._max_pitch_rate, self._max_pitch_rate))
        twist_ref[5] = float(np.clip(yaw_cmd, -self._max_yaw_rate, self._max_yaw_rate))

    def _set_linvel(self, joy):
        if not self.enable_linvel:
            self._current_twist_ref_world[0:3] = 0.0
            return
        twist_ref = self._current_twist_ref_world
        try:
            lx = float(joy.sticks[2])
            ly = float(joy.sticks[3])
        except Exception:
            lx, ly = 0.0, 0.0
        mag = float(np.hypot(lx, ly))
        if mag < self.dxy:
            self._v_magnitude = 0.0
        else:
            self._heading = np.arctan2(ly, lx) - math.pi/2.0
            norm_mag = min(mag, 1.0)
            self._v_magnitude = norm_mag * self._max_vxy_magn
        self._v_magnitude = float(np.clip(self._v_magnitude, a_min=0.0, a_max=self._max_vxy_magn))
        twist_ref[0] = self._v_magnitude * math.cos(self._heading)
        twist_ref[1] = self._v_magnitude * math.sin(self._heading)

    def _set_linvelz(self, joy):
        """
        If enable_linvelz is True, set vertical velocity (twist[2]) from triggers:
          twist[2] = -(rt_n - lt_n) * self._max_vz_magn
        where lt_n and rt_n are normalized triggers in [0,1] (same normalization used for yaw).
        If not enabled, this method does nothing (existing stepping/old logic remains elsewhere).
        """

        # defensive reads
        lt = float(joy.triggers[0])
        rt = float(joy.triggers[1])

        lt_n = self._norm_trigger(lt)
        rt_n = self._norm_trigger(rt)

        # same sign convention used for yaw earlier in this class: negative of (rt - lt)
        vz_cmd = (rt_n - lt_n) * float(self._max_vz_magn)

        # small deadzone
        if abs(vz_cmd) < 1e-6:
            vz_cmd = 0.0

        # only apply if linear velocity control is active (to avoid conflicts with pos mode)
        if self.enable_linvelz:
            # set z velocity directly (clipped)
            self._current_twist_ref_world[2] = float(np.clip(vz_cmd, -self._max_vz_magn, self._max_vz_magn))

    def _set_position(self, joy):
        if not self.enable_pos:
            robot_p = self._robot_state.root_state.get(data_type="p")[self.cluster_idx_np, :].reshape(-1)
            robot_p[2] = 0.0
            self._current_pos_ref[:] = robot_p

    def _write_to_shared_mem(self):
        self._shared_refs.rob_refs.root_state.synch_all(read=True)
        self._robot_state.root_state.synch_all(read=True, retry=True)

        if self.enable_pos:
            robot_p = self._robot_state.root_state.get(data_type="p")[self.cluster_idx_np, :].reshape(-1)
            robot_p[2] = 0.0
            self._shared_refs.rob_refs.root_state.set(data_type="p", data=self._current_pos_ref,
                                                    robot_idxs=self.cluster_idx_np)
            self._shared_refs.rob_refs.root_state.synch_retry(row_index=self.cluster_idx, col_index=0,
                                                            n_rows=1, n_cols=3, read=False)

        if self._agent_refs_world:
            if self.enable_omega:
                robot_q = self._robot_state.root_state.get(data_type="q")[self.cluster_idx_np, :].reshape(1, -1)
                world2base_frame_twist(t_w=self._current_twist_ref_world.reshape(1, -1),
                                       q_b=robot_q,
                                       t_out=self._current_twist_ref_base,
                                       omega=True,
                                       linvel=False)
            if self.enable_linvel:
                self._current_twist_ref_base[:, 0:3] = self._current_twist_ref_world.reshape(1, -1)[:, 0:3]
        else:
            self._current_twist_ref_base[:, :] = self._current_twist_ref_world.reshape(1, -1)

        self._shared_refs.rob_refs.root_state.set(data_type="twist", data=self._current_twist_ref_base,
                                                robot_idxs=self.cluster_idx_np)
        self._shared_refs.rob_refs.root_state.synch_retry(row_index=self.cluster_idx, col_index=7,
                                                        n_rows=1, n_cols=6, read=False)

    # -------------------
    # Input processing: detect edges and perform keyboard-like actions
    # -------------------
    def _process_joy_for_writes(self, joy):
        """
        Process joystick inputs and perform writes that replicate RefsFromKeyboard behavior
        with the new mapping rules requested by the user.

        Key/high-level behavior implemented here (summary):
         - Contacts use the left pad (hat): up,right,down,left -> contacts 0,1,2,3 respectively
         - Face buttons (X,B,A,Y) are reserved for mode selection and flight-mode controls
         - If flight-change mode is enabled (self._enable_flight_param_change) then
           omega/linvel/base-height modes are disabled. Otherwise, the three modes
           (omega, linvel, base-height) can be toggled with holds on face buttons
           (minimizing overlap with contact controls).
         - Left stick sets roll (x) and pitch (y) rates when omega is enabled
         - Triggers (LT,RT) are used to increase/decrease yaw reference (when omega enabled)
           and/or vertical velocity (when base-height mode + linvel enabled). In flight mode
           triggers increase/decrease the selected flight parameter for enabled legs.
         - Guide/menu button toggles an otherwise-unused phase-id-change flag (kept for parity)
         - Face buttons in flight mode: short-press toggles the contact-leg enable for the
           currently selected flight parameter; long-press (hold) selects which parameter
           is active: X=length, A=apex, Y=end. B is kept free for future use / minimal overlap.
        """
        # env index
        if self.env_index is not None:
            self.env_index.synch_all(read=True, retry=True)
            env_index = self.env_index.get_numpy_mirror()
            self._env_idx = env_index[0,0].item()
        self.cluster_idx = self._env_idx
        self.cluster_idx_np = np.array(self.cluster_idx)

        # Read current inputs
        cur_face = joy.face.copy()
        cur_hat = joy.hat.copy()
        cur_bumpers = joy.bumpers.copy()
        cur_trigs = np.array(joy.triggers, dtype=float)
        cur_stick_press = getattr(joy, 'stick_press', np.zeros(2, dtype=bool)).copy()

        # Mutual exclusion: if flight_change mode enabled, disable navigation/omega/height modes
        if getattr(self, '_enable_flight_param_change', False):
            # force disable other modes
            self.enable_omega = False
            self.enable_linvel = False
            # keep enable_pos as before, but base height will not be active in nav mode
        # otherwise the face buttons below may toggle them

        # 0) Guide/menu toggles a retained "phase id" flag (unused elsewhere but kept as requested)
        # back_start_home[2] is guide/home per JoyListenerZMQ mapping
        # cur_back_start_home = joy.back_start_home.copy()
        # if cur_back_start_home[2] and not self._prev_back_start_home[2]:
        #     # toggle enable_phase_id_change but it is otherwise unused
        #     self.enable_phase_id_change = not getattr(self, 'enable_phase_id_change', False)
        #     info = f"Phase ID change enabled (unused): {self.enable_phase_id_change}"
        #     Journal.log(self.__class__.__name__, "_set_phase_id", info, LogType.INFO, throw_when_excep=True)
        # self._prev_back_start_home = cur_back_start_home

        hx, hy = int(cur_hat[0]), int(cur_hat[1])
        # Build boolean per-direction
        hat_buttons = np.array([False, False, False, False], dtype=bool)  # up, right, down, left
        hat_buttons[0] = (hy > 0)
        hat_buttons[1] = (hx > 0)
        hat_buttons[2] = (hy < 0)
        hat_buttons[3] = (hx < 0)
        self._check_and_toggle("linvel", bool(hat_buttons[3]))
        self._check_and_toggle("linvelz", bool(hat_buttons[0]))
        self._check_and_toggle("omega", bool(hat_buttons[1]))
        self._check_and_toggle("pos", bool(hat_buttons[2]))

        # flight-change uses its own boolean to avoid confusion with _check_and_toggle map
        # We'll interpret a long press on Y to toggle flight-change mode
        # Manage flight-change hold detection similar to _check_and_toggle
        now = time.time()
        # flight-change hold tracking stored in self._flight_hold_since/_triggered
        # if not hasattr(self, '_flight_hold_since'):
        #     self._flight_hold_since = None
        #     self._flight_hold_triggered = False
        # if cur_face[3]:
        #     if self._flight_hold_since is None:
        #         self._flight_hold_since = now
        #     else:
        #         if (now - self._flight_hold_since) >= self.hold_time and not self._flight_hold_triggered:
        #             self._enable_flight_param_change = not getattr(self, '_enable_flight_param_change', False)
        #             info = f"Flight params change enabled: {self._enable_flight_param_change}"
        #             Journal.log(self.__class__.__name__, "_set_flight_params", info, LogType.INFO, throw_when_excep=True)
        #             self._flight_hold_triggered = True
        # else:
        #     self._flight_hold_since = None
        #     self._flight_hold_triggered = False

        # If flight mode enabled, ensure other modes are off (already enforced above)
        # if getattr(self, '_enable_flight_param_change', False):
        #     self.enable_omega = False
        #     self.enable_linvel = False
        #     self.enable_pos = False

        # 2) Contacts: use left pad (hat) for contact stepping; mapping kept
        #    hat up    -> contact 0 (left front) press/release
        #    hat right -> contact 1 (right front)
        #    hat down  -> contact 2 (left back)
        #    hat left  -> contact 3 (right back)
        # We detect edges on hat changes to simulate press/release behavior.
        # Build boolean per-direction
        hat_buttons = np.array([False, False, False, False], dtype=bool)  # up, right, down, left
        hat_buttons[0] = cur_face[3]
        hat_buttons[1] = cur_face[0]
        hat_buttons[2] = cur_face[2]
        hat_buttons[3] = cur_face[1]

        # map hat index order to contact indices preserving original ordering
        hat_to_contact = {0: 0, 1: 1, 2: 2, 3: 3}
        for hi in range(4):
            # rising edge -> press -> contact OFF
            if hat_buttons[hi] and not getattr(self, '_prev_hat_buttons', np.zeros(4, dtype=bool))[hi]:
                self._set_contacts(hat_to_contact[hi], is_contact=False)
            # falling edge -> release -> contact ON
            if not hat_buttons[hi] and getattr(self, '_prev_hat_buttons', np.zeros(4, dtype=bool))[hi]:
                self._set_contacts(hat_to_contact[hi], is_contact=True)
        self._prev_hat_buttons = hat_buttons.copy()

        # 3) Navigation / omega behaviour
        # Left stick sets roll (x) and pitch (y) of angular velocity (omega)
        # Triggers control yaw (z): RT positive, LT negative; yaw = (rt - lt) * max_yaw_rate
        # lsx = float(joy.sticks[0])
        # lsy = float(joy.sticks[1])

        # Read triggers (LT, RT). Some platforms give triggers in [-1,1] (rest -1, pressed +1),
        # others give [0,1] (rest 0, pressed 1). Normalize each to [0,1] then compute differential.

        # 4) Linear velocity and base-height behaviour
        # If base-height mode is enabled, and linvel is enabled -> control z velocity via triggers
        # otherwise in base-height mode and pos-mode active -> change position z
        # if getattr(self, 'enable_heightchange', False) and not getattr(self, '_enable_flight_param_change', False):
        # base-height active
        # try:
        #     lt = float(joy.triggers[0])
        #     rt = float(joy.triggers[1])
        # except Exception:
        #     lt, rt = 0.0, 0.0
        # vz_cmd = (rt - lt) * float(getattr(self, '_max_vz_magn', 0.5))
        # if self.enable_linvel:
        #     # set z velocity component
        #     self._current_twist_ref_world[2] = float(np.clip(vz_cmd, -self._max_vz_magn, self._max_vz_magn))
        # else:
        #     # adjust position target incrementally based on triggers
        #     # when RT pressed, increase z; when LT pressed, decrease z
        #     # we treat crossing of threshold as one-step change to avoid rapid drift
        #     thr = 0.5
        #     if (rt > thr) and (self._prev_triggers[1] <= thr):
        #         # increase position
        #         self._current_pos_ref[2] += self.height_dh
        #     if (lt > thr) and (self._prev_triggers[0] <= thr):
        #         self._current_pos_ref[2] -= self.height_dh
        # else:
        #     # not in base-height mode: if linvel disabled, zero vertical velocity
        #     if not self.enable_linvel and not getattr(self, '_enable_flight_param_change', False):
        #         self._current_twist_ref_world[2] = 0.0

        # 5) Flight-change mode handling (exclusive)
        # In flight mode: face buttons short-press toggle per-contact enable for the currently
        # selected parameter. Long-press on X/A/Y selects parameter: X=length, A=apex, Y=end.
        # if getattr(self, '_enable_flight_param_change', False):
        #     # initialize selected param storage
        #     if not hasattr(self, '_flight_param_selected'):
        #         self._flight_param_selected = 'length'  # default
        #     if not hasattr(self, '_flight_face_since'):
        #         self._flight_face_since = [None]*4
        #         self._flight_face_triggered = [False]*4

        #     # detect face button presses and durations
        #     for fi in range(4):
        #         if cur_face[fi] and self._flight_face_since[fi] is None:
        #             self._flight_face_since[fi] = time.time()
        #         if not cur_face[fi] and self._flight_face_since[fi] is not None:
        #             # button was released -> compute duration
        #             dur = time.time() - self._flight_face_since[fi]
        #             self._flight_face_since[fi] = None
        #             if dur >= self.hold_time:
        #                 # long press -> param select mapping for X(0)/A(2)/Y(3)
        #                 if fi == 0:
        #                     self._flight_param_selected = 'length'
        #                 elif fi == 2:
        #                     self._flight_param_selected = 'apex'
        #                 elif fi == 3:
        #                     self._flight_param_selected = 'end'
        #                 # B (fi==1) reserved / no long-press action
        #                 info = f"Flight param selected: {self._flight_param_selected}"
        #                 Journal.log(self.__class__.__name__, "_flight_param_select", info, LogType.INFO, throw_when_excep=True)
        #             else:
        #                 # short press -> toggle per-contact enable for this face index (map 0..3 -> contact 0..3)
        #                 contact_idx = fi
        #                 self._d_fparam_enabled_contact_i[contact_idx] = not self._d_fparam_enabled_contact_i[contact_idx]
        #                 info = f"Flight param enable for contact {contact_idx}: {self._d_fparam_enabled_contact_i[contact_idx]}"
        #                 Journal.log(self.__class__.__name__, "_set_flight_params", info, LogType.INFO, throw_when_excep=True)

        #     # triggers act as +/- for the selected param across enabled contacts
        #     thr = 0.1
        #     # RT increments proportional to its value when above threshold
        #     if cur_trigs[1] > thr and self._prev_triggers[1] <= thr:
        #         # RT just crossed -> perform increment for all enabled contacts
        #         for ci in range(4):
        #             if self._d_fparam_enabled_contact_i[ci]:
        #                 self._update_flight_params(contact_idx=ci, increment=True)
        #     if cur_trigs[0] > thr and self._prev_triggers[0] <= thr:
        #         for ci in range(4):
        #             if self._d_fparam_enabled_contact_i[ci]:
        #                 self._update_flight_params(contact_idx=ci, increment=False)

        # 6) update prev triggers snapshot for edge detection used above
        self._prev_triggers = cur_trigs

        # end of processing; other bits (writing of root_state) happen in the external loop

    # -------------------
    # Main run loop: listen to JoyListenerZMQ and write at ~100 Hz
    # -------------------
    # def run(self, connect: str, topic: str, poll_interval: float = 0.01):
    #     info = f"Ready. Starting to listen for joystick commands..."
    #     Journal.log(self.__class__.__name__, "run", info, LogType.INFO, throw_when_excep=True)

    #     # start listener
    #     joy_listener = JoyListenerZMQ(connect=connect, topic=topic, poll_interval=poll_interval)
    #     joy_listener.start()

    #     try:
    #         # main loop
    #         while not joy_listener.done:
    #             # synchronize cluster/env index and read joy state to perform writes
    #             self._process_joy_for_writes(joy_listener)
    #             # compute twist/pos references like RefsFromJoy
    #             # first, update the high level twist world vector from joystick
    #             self._set_omega(joy_listener)
    #             self._set_linvel(joy_listener)
    #             self._set_position(joy_listener)
    #             # then write to shared mems
    #             self._write_to_shared_mem()
    #             time.sleep(0.01)
    #     except KeyboardInterrupt:
    #         print("[RefsFromJoy][run]: Exiting...")
    #     finally:
    #         joy_listener.stop()
    #         self._close()

    def run(self, connect: str, topic: str, poll_interval: float = 0.01,
            callback: Optional[Callable[[Any, Any], None]] = None, callback_arg: Any = None):
        """
        Main run loop for RefsFromJoy.

        Parameters
        ----------
        connect : str
            ZeroMQ connect address (host:port) for JoyListenerZMQ.
        topic : str
            ZMQ topic to subscribe to.
        poll_interval : float
            Poll interval for the JoyListenerZMQ.
        callback : Optional[Callable[[joy_listener, callback_arg], None]]
            Optional callback invoked every loop with the running joy_listener and the provided callback_arg.
            The callback is NOT expected to return anything; if it wants to request shutdown it should
            set external flags (e.g. via callback_arg shared wrapper).
        callback_arg : Any
            Arbitrary object passed through to the callback (e.g. safety_flag wrapper).
        """
        info = f"Ready. Starting to listen for joystick commands..."
        Journal.log(self.__class__.__name__, "run", info, LogType.INFO, throw_when_excep=True)

        # start listener
        joy_listener = JoyListenerZMQ(connect=connect, topic=topic, poll_interval=poll_interval)
        joy_listener.start()

        # main loop
        while not joy_listener.done:
            # optional external callback (e.g. remote-exit check) - DO NOT expect return value
            if callback is not None:
                # give callback access to both the live listener and the extra arg
                ret=callback(joy_listener, callback_arg)
                if not ret: 
                    break

            # synchronize env/cluster index and process joystick-driven writes
            self._process_joy_for_writes(joy_listener)

            # compute twist/pos references like RefsFromJoy
            self._set_omega(joy_listener)
            self._set_linvel(joy_listener)
            self._set_linvelz(joy_listener)
            self._set_position(joy_listener)

            # then write to shared memory
            self._write_to_shared_mem()

            # poll interval
            time.sleep(poll_interval)
        
        print("[RefsFromJoy][run]: Exiting...")
        joy_listener.stop()
        self._close()

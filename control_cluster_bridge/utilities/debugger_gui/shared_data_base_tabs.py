from PyQt5.QtWidgets import QWidget

from control_cluster_bridge.utilities.debugger_gui.gui_exts import SharedDataWindow
from control_cluster_bridge.utilities.debugger_gui.plot_utils import RtPlotWindow

from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState, RhcCmds
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcInternal
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcStatus
from control_cluster_bridge.utilities.shared_data.sim_data import SharedSimInfo
from control_cluster_bridge.utilities.shared_data.cluster_profiling import RhcProfiling
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcRefs

from SharsorIPCpp.PySharsorIPC import VLevel

from control_cluster_bridge.utilities.debugger_gui.plot_utils import WidgetUtils

import numpy as np

class FullRobStateWindow(SharedDataWindow):

    def __init__(self, 
            shared_mem_client,
            update_data_dt: int,
            update_plot_dt: int,
            window_duration: int,
            window_buffer_factor: int = 2,
            namespace = "",
            name="",
            parent: QWidget = None, 
            verbose = False):
        
        self._shared_mem_client = shared_mem_client

        super().__init__(update_data_dt = update_data_dt,
            update_plot_dt = update_plot_dt,
            window_duration = window_duration,
            grid_n_rows = 5,
            grid_n_cols = 2,
            window_buffer_factor = window_buffer_factor,
            namespace = namespace,
            name = name,
            parent = parent, 
            verbose = verbose)

    def _initialize(self):

        self.rt_plotters.append(RtPlotWindow(data_dim=self.shared_data_clients[0].root_state.get(data_type="p").shape[1],
                    n_data = 1,
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Root position", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=["p_x", "p_y", "p_z"], 
                    ylabel="[m]"))
        
        self.rt_plotters.append(RtPlotWindow(
                    data_dim=self.shared_data_clients[0].root_state.get(data_type="q").shape[1],
                    n_data = 1,
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Root orientation", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=["q_w", "q_i", "q_j", "q_k"]))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=self.shared_data_clients[0].root_state.get(data_type="v").shape[1],
                    n_data = 1,
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt, 
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Base linear vel.", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=["v_x", "v_y", "v_z"], 
                    ylabel="[m/s]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=self.shared_data_clients[0].root_state.get(data_type="omega").shape[1],
                    n_data = 1, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt, 
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Base angular vel.",
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=["omega_x", "omega_y", "omega_z"], 
                    ylabel="[rad/s]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=self.shared_data_clients[0].n_jnts(),
                    n_data = 1,
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Joints q",
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=self.shared_data_clients[0].jnt_names(), 
                    ylabel="[rad]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=self.shared_data_clients[0].n_jnts(),
                    n_data = 1, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Joints v",
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=self.shared_data_clients[0].jnt_names(), 
                    ylabel="[rad/s]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=self.shared_data_clients[0].n_jnts(),
                    n_data = 1, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Joints a",
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=self.shared_data_clients[0].jnt_names(), 
                    ylabel="[rad/s^2]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=self.shared_data_clients[0].n_jnts(),
                    n_data = 1,
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Joints efforts",
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=self.shared_data_clients[0].jnt_names(), 
                    ylabel="[Nm]"))
        
        contact_wrench_legend = [""] * self.shared_data_clients[0].contact_wrenches.n_cols
        contact_names = self.shared_data_clients[0].contact_names()

        for i in range(self.shared_data_clients[0].n_contacts()):

            contact_wrench_legend[i * 3] = "f_x - " + contact_names[i]
            contact_wrench_legend[i * 3 + 1] = "f_y - " + contact_names[i]
            contact_wrench_legend[i * 3 + 2] = "f_z - " + contact_names[i]

        for i in range(self.shared_data_clients[0].n_contacts()):

            contact_wrench_legend[i * 3 + 3 * len(contact_names)] = "t_x - " + contact_names[i]
            contact_wrench_legend[i * 3 + 1 + 3 * len(contact_names)] = "t_y - " + contact_names[i]
            contact_wrench_legend[i * 3 + 2 + 3 * len(contact_names)] = "t_z - " + contact_names[i]

        self.rt_plotters.append(RtPlotWindow(data_dim=len(contact_wrench_legend),
                    n_data = 1, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Contact wrenches",
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=contact_wrench_legend, 
                    ylabel="[N] - [Nm]"))
        
        # root state
        self.grid.addFrame(self.rt_plotters[0].base_frame, 0, 0)
        self.grid.addFrame(self.rt_plotters[1].base_frame, 0, 1)
        self.grid.addFrame(self.rt_plotters[2].base_frame, 1, 0)
        self.grid.addFrame(self.rt_plotters[3].base_frame, 1, 1)
        
        # joint state
        self.grid.addFrame(self.rt_plotters[4].base_frame, 2, 0)
        self.grid.addFrame(self.rt_plotters[5].base_frame, 2, 1)
        self.grid.addFrame(self.rt_plotters[6].base_frame, 3, 0)
        self.grid.addFrame(self.rt_plotters[7].base_frame, 3, 1)

        # contact state
        self.grid.addFrame(self.rt_plotters[8].base_frame, 4, 0)

    def _init_shared_data(self):
        
        self.shared_data_clients.append(self._shared_mem_client)
        self.shared_data_clients[0].run()
            
    def _post_shared_init(self):
        
        pass

    def update(self,
            index: int):

        if not self._terminated:
            
            # update from shared mem
            self.shared_data_clients[0].synch_from_shared_mem()
            np_idx = np.array(index)

            # root state
            self.rt_plotters[0].rt_plot_widget.update(self.shared_data_clients[0].root_state.get(data_type="p", robot_idxs=np_idx).flatten())
            self.rt_plotters[1].rt_plot_widget.update(self.shared_data_clients[0].root_state.get(data_type="q", robot_idxs=np_idx).flatten())
            self.rt_plotters[2].rt_plot_widget.update(self.shared_data_clients[0].root_state.get(data_type="v", robot_idxs=np_idx).flatten())
            self.rt_plotters[3].rt_plot_widget.update(self.shared_data_clients[0].root_state.get(data_type="omega", robot_idxs=np_idx).flatten())

            # joint state
            self.rt_plotters[4].rt_plot_widget.update(self.shared_data_clients[0].jnts_state.get(data_type="q",robot_idxs=np_idx).flatten())
            self.rt_plotters[5].rt_plot_widget.update(self.shared_data_clients[0].jnts_state.get(data_type="v",robot_idxs=np_idx).flatten())
            self.rt_plotters[6].rt_plot_widget.update(self.shared_data_clients[0].jnts_state.get(data_type="a",robot_idxs=np_idx).flatten())
            self.rt_plotters[7].rt_plot_widget.update(self.shared_data_clients[0].jnts_state.get(data_type="eff",robot_idxs=np_idx).flatten())

            # contact state
            self.rt_plotters[8].rt_plot_widget.update(self.shared_data_clients[0].contact_wrenches.get(data_type="w", robot_idxs=np_idx).flatten())

class RobotStates(FullRobStateWindow):

    def __init__(self,
            update_data_dt: int,
            update_plot_dt: int,
            window_duration: int,
            window_buffer_factor: int = 2,
            namespace = "",
            parent: QWidget = None, 
            verbose = False):

        name = "RobotStates"

        robot_state = RobotState(namespace=namespace,
                                    is_server=False, 
                                    with_gpu_mirror=False, 
                                    safe=False,
                                    verbose=verbose,
                                    vlevel=VLevel.V2)
        
        super().__init__(shared_mem_client=robot_state,
            update_data_dt=update_data_dt,
            update_plot_dt=update_plot_dt,
            window_duration=window_duration,
            window_buffer_factor=window_buffer_factor,
            namespace=namespace,
            name=name,
            parent=parent, 
            verbose=verbose)

class RHCmds(FullRobStateWindow):

    def __init__(self,
            update_data_dt: int,
            update_plot_dt: int,
            window_duration: int,
            window_buffer_factor: int = 2,
            namespace = "",
            parent: QWidget = None, 
            verbose = False):

        name = "Rhcmds"

        rhc_cmds = RhcCmds(namespace=namespace,
                        is_server=False, 
                        with_gpu_mirror=False, 
                        safe=False,
                        verbose=verbose,
                        vlevel=VLevel.V2)
                                            
        super().__init__(shared_mem_client=rhc_cmds,
            update_data_dt=update_data_dt,
            update_plot_dt=update_plot_dt,
            window_duration=window_duration,
            window_buffer_factor=window_buffer_factor,
            namespace=namespace,
            name=name,
            parent=parent, 
            verbose=verbose)

class RHCRefs(SharedDataWindow):

    def __init__(self, 
            update_data_dt: int,
            update_plot_dt: int,
            window_duration: int,
            window_buffer_factor: int = 2,
            namespace = "",
            parent: QWidget = None, 
            verbose = False):
        
        name = "RhcRefs"

        super().__init__(update_data_dt = update_data_dt,
            update_plot_dt = update_plot_dt,
            window_duration = window_duration,
            grid_n_rows = 5,
            grid_n_cols = 2,
            window_buffer_factor = window_buffer_factor,
            namespace = namespace,
            name = name,
            parent = parent, 
            verbose = verbose)

    def _initialize(self):

        jnt_ref_names = [element + "_ref" for element in self.shared_data_clients[0].rob_refs.jnt_names()]

        jnt_leg_full = jnt_ref_names + self.shared_data_clients[0].rob_refs.jnt_names()

        self.rt_plotters.append(RtPlotWindow(data_dim=2 * self.shared_data_clients[0].rob_refs.root_state.get(data_type="p").shape[1],
                    n_data = 1,
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Root ref VS meas. position", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=["p_x_ref", "p_y_ref", "p_z_ref",
                            "p_x", "p_y", "p_z"], 
                    ylabel="[m]"))
        
        self.rt_plotters.append(RtPlotWindow(
                    data_dim=2 * self.shared_data_clients[0].rob_refs.root_state.get(data_type="q").shape[1],
                    n_data = 1,
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Root ref VS meas. orientation", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=["q_w_ref", "q_i_ref", "q_j_ref", "q_k_ref",
                            "q_w", "q_i", "q_j", "q_k"]))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=2 * self.shared_data_clients[0].rob_refs.root_state.get(data_type="v").shape[1],
                    n_data = 1,
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt, 
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Root ref VS meas. linear vel.", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=["v_x_ref", "v_y_ref", "v_z_ref",
                            "v_x", "v_y", "v_z"], 
                    ylabel="[m/s]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=2 * self.shared_data_clients[0].rob_refs.root_state.get(data_type="omega").shape[1],
                    n_data = 1, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt, 
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Root ref. VS meas. angular vel.",
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=["omega_x_ref", "omega_y_ref", "omega_z_ref",
                            "omega_x", "omega_y", "omega_z"], 
                    ylabel="[rad/s]"))
    
        self.rt_plotters.append(RtPlotWindow(data_dim=self.shared_data_clients[0].contact_flags.n_cols,
                    n_data = 1, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt, 
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Contact flags",
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=self.shared_data_clients[0].rob_refs.contact_names(), 
                    ylabel="[bool]"))

        self.rt_plotters.append(RtPlotWindow(data_dim=1,
                    n_data = 1, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt, 
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Phase ID",
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=[""], 
                    ylabel="[int]"))
        
        # root state
        self.grid.addFrame(self.rt_plotters[0].base_frame, 0, 0)
        self.grid.addFrame(self.rt_plotters[1].base_frame, 0, 1)
        self.grid.addFrame(self.rt_plotters[2].base_frame, 1, 0)
        self.grid.addFrame(self.rt_plotters[3].base_frame, 1, 1)
        self.grid.addFrame(self.rt_plotters[4].base_frame, 2, 0)
        self.grid.addFrame(self.rt_plotters[5].base_frame, 2, 1)
        
    def _init_shared_data(self):
        
        self.shared_data_clients.append(RhcRefs(namespace=self.namespace,
                                    is_server=False,
                                    with_gpu_mirror=False, 
                                    safe=False,
                                    verbose=self.verbose,
                                    vlevel=VLevel.V2))
        
        self.shared_data_clients.append(RobotState(namespace=self.namespace,
                                    is_server=False,
                                    with_gpu_mirror=False, 
                                    safe=False,
                                    verbose=self.verbose,
                                    vlevel=VLevel.V2))
        
        self.shared_data_clients[0].run()
        self.shared_data_clients[1].run()

    def _post_shared_init(self):
        
        pass

    def update(self,
            index: int):

        if not self._terminated:
            
            # update from shared mem
            self.shared_data_clients[0].rob_refs.synch_from_shared_mem()
            self.shared_data_clients[0].contact_flags.synch_all(read=True, retry=True)
            self.shared_data_clients[0].phase_id.synch_all(read=True, retry=True)

            self.shared_data_clients[1].synch_from_shared_mem()

            np_idx = np.array(index)

            p_ref = self.shared_data_clients[0].rob_refs.root_state.get(data_type="p",robot_idxs=np_idx).flatten()
            p_meas = self.shared_data_clients[1].root_state.get(data_type="p",robot_idxs=np_idx).flatten()
            p_full = np.concatenate((p_ref, p_meas)) 
                        
            q_ref = self.shared_data_clients[0].rob_refs.root_state.get(data_type="q",robot_idxs=np_idx).flatten()
            q_meas = self.shared_data_clients[1].root_state.get(data_type="q",robot_idxs=np_idx).flatten()
            q_full = np.concatenate((q_ref, q_meas)) 

            v_ref = self.shared_data_clients[0].rob_refs.root_state.get(data_type="v",robot_idxs=np_idx).flatten()
            v_meas = self.shared_data_clients[1].root_state.get(data_type="v",robot_idxs=np_idx).flatten()
            v_full = np.concatenate((v_ref, v_meas)) 

            omega_ref = self.shared_data_clients[0].rob_refs.root_state.get(data_type="omega",robot_idxs=np_idx).flatten()
            omega_meas = self.shared_data_clients[1].root_state.get(data_type="omega",robot_idxs=np_idx).flatten()
            omega_full = np.concatenate((omega_ref, omega_meas)) 

            # # root state
            self.rt_plotters[0].rt_plot_widget.update(p_full)
            self.rt_plotters[1].rt_plot_widget.update(q_full)
            self.rt_plotters[2].rt_plot_widget.update(v_full)
            self.rt_plotters[3].rt_plot_widget.update(omega_full)

            contact_flags = self.shared_data_clients[0].contact_flags.get_numpy_mirror()
            phase_id = self.shared_data_clients[0].phase_id.get_numpy_mirror()
            self.rt_plotters[4].rt_plot_widget.update(contact_flags[index, :])
            self.rt_plotters[5].rt_plot_widget.update(phase_id[index, :])

class RHCInternal(SharedDataWindow):

    def __init__(self, 
        name: str,
        update_data_dt: int,
        update_plot_dt: int,
        window_duration: int,
        window_buffer_factor: int = 2,
        namespace = "",
        parent: QWidget = None, 
        verbose = False,
        is_cost: bool = True,
        is_constraint: bool = False,
        add_settings_tab = True,
        settings_title = "SETTINGS (RHCInternal)"
        ):
                
        self.is_cost = is_cost
        self.is_constraint = is_constraint

        self.cluster_size = -1

        self.current_node_index = 0

        self.names = []
        self.dims = []
        self.n_nodes = -1

        self.name = name
        
        self.enable_data = True
        self._contact_names = None

        super().__init__(update_data_dt = update_data_dt,
            update_plot_dt = update_plot_dt,
            window_duration = window_duration,
            window_buffer_factor = window_buffer_factor,
            grid_n_rows = 2,
            grid_n_cols = 3,
            namespace = namespace,
            name = name,
            parent = parent, 
            verbose = verbose,
            add_settings_tab = add_settings_tab,
            settings_title = settings_title
            )

    def _init_shared_data(self):
        
        is_server = False
        
        self.rhc_status_info = RhcStatus(is_server=is_server,
                            namespace=self.namespace, 
                            verbose=self.verbose,
                            vlevel=VLevel.V1)

        self.rhc_status_info.run()
        self.rhc_status_info.close()

        self.cluster_size = self.rhc_status_info.trigger.n_rows

        enable_costs = False
        enable_constr = False

        if self.is_cost:
            
            enable_costs = True

        if self.is_constraint:
            
            enable_constr = True

        if self.is_cost or self.is_constraint:

            self.enable_data = False

        config = RhcInternal.Config(is_server=is_server, 
                        enable_q=self.enable_data, 
                        enable_v=self.enable_data, 
                        enable_a=self.enable_data, 
                        enable_a_dot=False, 
                        enable_f=self.enable_data,
                        enable_f_dot=False, 
                        enable_eff=False,
                        enable_costs=enable_costs, 
                        enable_constr=enable_constr)

        # view of rhc internal data
        for i in range(0, self.cluster_size):

            self.shared_data_clients.append(RhcInternal(config=config,
                                            namespace=self.namespace,
                                            rhc_index = i,
                                            verbose=self.verbose,
                                            vlevel=VLevel.V2,
                                            safe=False))
        
        # run clients
        for i in range(0, self.cluster_size):

            self.shared_data_clients[i].run()
        
        if self.enable_data:

            rhc_refs = RhcRefs(namespace=self.namespace,
                                    is_server=False,
                                    with_gpu_mirror=False, 
                                    safe=False,
                                    verbose=self.verbose,
                                    vlevel=VLevel.V2)
            
            rhc_refs.run()

            self._contact_names = rhc_refs.rob_refs.contact_names()

            rhc_refs.close() # not needed anymore

    def _post_shared_init(self):
        
        if not self.enable_data:

            if self.is_cost:

                self.names = self.shared_data_clients[0].costs.names
                self.dims = self.shared_data_clients[0].costs.dimensions
                self.n_nodes =  self.shared_data_clients[0].costs.n_nodes

            if self.is_constraint:

                self.names = self.shared_data_clients[0].cnstr.names
                self.dims = self.shared_data_clients[0].cnstr.dimensions
                self.n_nodes =  self.shared_data_clients[0].cnstr.n_nodes

            # import math 

            # grid_size = math.ceil(math.sqrt(len(self.names)))

            # # distributing plots over a square grid
            # self.grid_n_rows = grid_size
            # self.grid_n_cols = grid_size

            self.grid_n_rows = len(self.names)

            self.grid_n_cols = 1
        
        else:

            self.grid_n_rows = 4
            self.grid_n_cols = 2

    def _initialize(self):
        
        if not self.enable_data:

            base_name = ""

            if self.is_cost:

                base_name = "Cost name"

            if self.is_constraint:

                base_name = "Constraint name"

            # distribute plots on each row
            counter = 0
            for i in range(0, self.grid_n_rows):
                
                for j in range(0, self.grid_n_cols):
                
                    if (counter < len(self.names)):
                        
                        legend_list = [""] * self.dims[counter]
                        for k in range(self.dims[counter]):
                            
                            legend_list[k] = str(k)

                        self.rt_plotters.append(RtPlotWindow(data_dim=self.dims[counter],
                                    n_data = self.n_nodes,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"{base_name}: {self.names[counter]}", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=legend_list, 
                                    ylabel=""))

                        self.grid.addFrame(self.rt_plotters[counter].base_frame, i, j)

                        counter = counter + 1
        
        else:
            
            n_dims_q = self.shared_data_clients[0].q.n_rows 
            n_nodes_q = self.shared_data_clients[0].q.n_cols
            legend = [""] * n_dims_q
            for i in range(len(legend)):
                legend[i] = str(i)
            q_legend = ["q_" + element for element in legend]

            n_dims_v = self.shared_data_clients[0].v.n_rows 
            n_nodes_v = self.shared_data_clients[0].v.n_cols
            legend = [""] * n_dims_v
            for i in range(len(legend)):
                legend[i] = str(i)
            v_legend = ["v_" + element for element in legend]
            
            n_dims_a = self.shared_data_clients[0].a.n_rows 
            n_nodes_a = self.shared_data_clients[0].a.n_cols
            legend = [""] * n_dims_a
            for i in range(len(legend)):
                legend[i] = str(i)            
            a_legend = ["a_" + element for element in legend]

            n_dims_f = self.shared_data_clients[0].f.n_rows 
            n_nodes_f = self.shared_data_clients[0].f.n_cols
            f_legend_base_f = ["f_x", "f_y", "f_z"]
            f_legend_base_t = ["t_x", "t_y", "t_z"]
            force_legend = f_legend_base_f * len(self._contact_names)
            torque_legend = f_legend_base_t * len(self._contact_names)
            legend_wrench = force_legend + torque_legend
            for i in range(len(self._contact_names)):
                contact_name = self._contact_names[i]
                for j in range(3):
                    legend_wrench[3 * i + j] = legend_wrench[3 * i + j] + "_" + contact_name
                    legend_wrench[3 * len(self._contact_names) + 3 * i + j] = legend_wrench[3 * len(self._contact_names) +3 * i + j] + "_" + contact_name                    

            self.rt_plotters.append(RtPlotWindow(data_dim=n_dims_q,
                                    n_data = n_nodes_q,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Rhc internal data: q", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=q_legend, 
                                    ylabel=""))
            
            self.rt_plotters.append(RtPlotWindow(data_dim=n_dims_v,
                                    n_data = n_nodes_v,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Rhc internal data: v", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=v_legend, 
                                    ylabel=""))
            
            self.rt_plotters.append(RtPlotWindow(data_dim=n_dims_a,
                                    n_data = n_nodes_a,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Rhc internal data: a", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=a_legend, 
                                    ylabel=""))

            self.rt_plotters.append(RtPlotWindow(data_dim=n_dims_f,
                                    n_data = n_nodes_f,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Rhc internal data: f", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=legend_wrench, 
                                    ylabel=""))

            self.grid.addFrame(self.rt_plotters[0].base_frame, 0, 0)
            self.grid.addFrame(self.rt_plotters[1].base_frame, 0, 1)
            self.grid.addFrame(self.rt_plotters[2].base_frame, 1, 0)
            self.grid.addFrame(self.rt_plotters[3].base_frame, 1, 1)

            self.n_nodes = n_nodes_q # we assume all data to be of same dimension (even if not defined on some nodes)

    def _finalize_grid(self):
        
        widget_utils = WidgetUtils()

        settings_frames = []

        node_index_slider = widget_utils.generate_complex_slider(
                        parent=None, 
                        parent_layout=None,
                        min_shown=f"{0}", min= 0, 
                        max_shown=f"{self.n_nodes - 1}", 
                        max=self.n_nodes - 1, 
                        init_val_shown=f"{0}", init=0, 
                        title="node index slider", 
                        callback=self._update_node_idx)
        
        settings_frames.append(node_index_slider)
        
        self.grid.addToSettings(settings_frames)
    
    def _update_node_idx(self,
                    idx: int):

        self.current_node_index = idx

        self.grid.settings_widget_list[0].current_val.setText(f'{idx}')

        for i in range(0, len(self.rt_plotters)):
            
            self.rt_plotters[i].rt_plot_widget.switch_to_data(data_idx = self.current_node_index)

    def update(self,
            index: int):

        self.shared_data_clients[index].synch()

        if not self._terminated:
            
            if not self.enable_data:

                for i in range(0, len(self.names)):
                    
                    # iterate over data names (i.e. plots)

                        if self.is_cost:
                            data = np.atleast_2d(self.shared_data_clients[index].read_cost(self.names[i])[:, :])
                            self.rt_plotters[i].rt_plot_widget.update(data)

                        if self.is_constraint:
                            data = np.atleast_2d(self.shared_data_clients[index].read_constr(self.names[i])[:, :])
                            self.rt_plotters[i].rt_plot_widget.update(data)

            else:
                
                self.rt_plotters[0].rt_plot_widget.update(self.shared_data_clients[0].q.get_numpy_mirror()[:, :])
                self.rt_plotters[1].rt_plot_widget.update(self.shared_data_clients[0].v.get_numpy_mirror()[:, :])
                self.rt_plotters[2].rt_plot_widget.update(self.shared_data_clients[0].a.get_numpy_mirror()[:, :])
                self.rt_plotters[3].rt_plot_widget.update(self.shared_data_clients[0].f.get_numpy_mirror()[:, :])

class SimInfo(SharedDataWindow):

    def __init__(self, 
        update_data_dt: int,
        update_plot_dt: int,
        window_duration: int,
        window_buffer_factor: int = 2,
        namespace = "",
        parent: QWidget = None, 
        verbose = False,
        add_settings_tab = False,
        settings_title = " Latest values"
        ):
        
        name = "SimInfo"

        super().__init__(update_data_dt = update_data_dt,
            update_plot_dt = update_plot_dt,
            window_duration = window_duration,
            window_buffer_factor = window_buffer_factor,
            grid_n_rows = 1,
            grid_n_cols = 1,
            namespace = namespace,
            name = name,
            parent = parent, 
            verbose = verbose,
            add_settings_tab = add_settings_tab,
            settings_title = settings_title
            )

    def _init_shared_data(self):
        
        is_server = False
        
        self.shared_data_clients.append(SharedSimInfo(
                                            namespace=self.namespace,
                                            is_server=is_server,
                                            safe=False))
        
        self.shared_data_clients[0].run()

    def _post_shared_init(self):
        
        self.grid_n_rows = 1

        self.grid_n_cols = 1

    def _initialize(self):
        
        base_name = "SharedSimInfo"
        
        self.rt_plotters.append(RtPlotWindow(data_dim=len(self.shared_data_clients[0].param_keys),
                    n_data = 1,
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name=f"{base_name}", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=self.shared_data_clients[0].param_keys, 
                    ylabel=""))

        self.grid.addFrame(self.rt_plotters[0].base_frame, 0, 0)

    def _finalize_grid(self):
        
        if self.add_settings_tab:

            widget_utils = WidgetUtils()

            settings_frames = []

            # sim_info_widget = widget_utils.create_scrollable_label_list(parent=None, 
            #                                     parent_layout=None, 
            #                                     list_names=self.shared_data_clients[0].param_keys,
            #                                     title="RT SIMULATOR INFO", 
            #                                     init=[np.nan] * len(self.shared_data_clients[0].param_keys))
            
            # settings_frames.append(sim_info_widget)
            
            self.grid.addToSettings(settings_frames)

    def update(self,
            index: int):

        # index not used here (no dependency on cluster index)

        if not self._terminated:
            
            data = self.shared_data_clients[0].get().flatten()

            # updates side data
            # self.grid.settings_widget_list[0].update(data)

            # updates rt plot
            self.rt_plotters[0].rt_plot_widget.update(data)

class RHCProfiling(SharedDataWindow):

    def __init__(self, 
        update_data_dt: int,
        update_plot_dt: int,
        window_duration: int,
        window_buffer_factor: int = 2,
        namespace = "",
        parent: QWidget = None, 
        verbose = False,
        add_settings_tab = False,
        settings_title = " Latest values"
        ):
        
        name = "RhcProfiling"

        super().__init__(update_data_dt = update_data_dt,
            update_plot_dt = update_plot_dt,
            window_duration = window_duration,
            window_buffer_factor = window_buffer_factor,
            grid_n_rows = 1,
            grid_n_cols = 1,
            namespace = namespace,
            name = name,
            parent = parent, 
            verbose = verbose,
            add_settings_tab = add_settings_tab,
            settings_title = settings_title
            )

    def _init_shared_data(self):
        
        is_server = False
        
        self.shared_data_clients.append(RhcProfiling(is_server=is_server,
                                            name=self.namespace, 
                                            verbose=True, 
                                            vlevel=VLevel.V2,
                                            safe=True))
        
        self.shared_data_clients[0].run()

    def _post_shared_init(self):
        
        self.grid_n_rows = 3

        self.grid_n_cols = 2

    def _initialize(self):
        
        cluster_size = self.shared_data_clients[0].cluster_size

        cluster_idx_legend = [""] * cluster_size

        for i in range(cluster_size):
            
            cluster_idx_legend[i] = str(i)
        
        self.rt_plotters.append(RtPlotWindow(data_dim=len(self.shared_data_clients[0].param_keys),
                                    n_data = 1,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Cluster cumulative data", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=self.shared_data_clients[0].param_keys, 
                                    ylabel="[s]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=cluster_size,
                                    n_data = 1,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Problem update dt", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=cluster_idx_legend, 
                                    ylabel="[s]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=cluster_size,
                                    n_data = 1,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Phases shift dt", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=cluster_idx_legend, 
                                    ylabel="[s]"))

        self.rt_plotters.append(RtPlotWindow(data_dim=cluster_size,
                                    n_data = 1,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Task ref. update dt", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=cluster_idx_legend, 
                                    ylabel="[s]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=cluster_size,
                                    n_data = 1,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Rti solution time", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=cluster_idx_legend, 
                                    ylabel="[s]"))

        self.rt_plotters.append(RtPlotWindow(data_dim=cluster_size,
                                    n_data = 1,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Cumulative solve dt", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=cluster_idx_legend, 
                                    ylabel="[s]"))
        
        self.grid.addFrame(self.rt_plotters[0].base_frame, 0, 0)
        self.grid.addFrame(self.rt_plotters[1].base_frame, 0, 1)
        self.grid.addFrame(self.rt_plotters[2].base_frame, 1, 0)
        self.grid.addFrame(self.rt_plotters[3].base_frame, 1, 1)
        self.grid.addFrame(self.rt_plotters[4].base_frame, 2, 0)
        self.grid.addFrame(self.rt_plotters[5].base_frame, 2, 1)

    def _finalize_grid(self):
        
        if self.add_settings_tab:

            widget_utils = WidgetUtils()

            settings_frames = []

            # sim_info_widget = widget_utils.create_scrollable_label_list(parent=None, 
            #                                     parent_layout=None, 
            #                                     list_names=self.shared_data_clients[0].param_keys,
            #                                     title="RT SIMULATOR INFO", 
            #                                     init=[np.nan] * len(self.shared_data_clients[0].param_keys))
            
            # settings_frames.append(sim_info_widget)
            
            self.grid.addToSettings(settings_frames)

    def update(self,
            index: int):

        # index not used here (no dependency on cluster index)

        if not self._terminated:
            
            # get cumulative data
            data = self.shared_data_clients[0].get_all_info().flatten()
            self.rt_plotters[0].rt_plot_widget.update(data)

            # prb update
            self.shared_data_clients[0].prb_update_dt.synch_all(read = True, 
                                                        retry=False)
            self.rt_plotters[1].rt_plot_widget.update(self.shared_data_clients[0].prb_update_dt.get_numpy_mirror())
            
            # phase shift
            self.shared_data_clients[0].phase_shift_dt.synch_all(read = True, 
                                                        retry=False)
            self.rt_plotters[2].rt_plot_widget.update(self.shared_data_clients[0].phase_shift_dt.get_numpy_mirror())

            # task ref update
            self.shared_data_clients[0].task_ref_update_dt.synch_all(read = True, 
                                                        retry=False)
            self.rt_plotters[3].rt_plot_widget.update(self.shared_data_clients[0].task_ref_update_dt.get_numpy_mirror())
            
            # rti sol time
            self.shared_data_clients[0].rti_sol_time.synch_all(read = True, 
                                                        retry=False)
            self.rt_plotters[4].rt_plot_widget.update(self.shared_data_clients[0].rti_sol_time.get_numpy_mirror())

            # whole solve loop
            self.shared_data_clients[0].solve_loop_dt.synch_all(read = True, 
                                                        retry=False)
            self.rt_plotters[5].rt_plot_widget.update(self.shared_data_clients[0].solve_loop_dt.get_numpy_mirror())
            
            # updates side data
            # self.grid.settings_widget_list[0].update(data)

class RHCStatus(SharedDataWindow):

    def __init__(self, 
        update_data_dt: int,
        update_plot_dt: int,
        window_duration: int,
        window_buffer_factor: int = 2,
        namespace = "",
        parent: QWidget = None, 
        verbose = False,
        add_settings_tab = True,
        ):
        
        name = "RhcStatus"

        super().__init__(update_data_dt = update_data_dt,
            update_plot_dt = update_plot_dt,
            window_duration = window_duration,
            window_buffer_factor = window_buffer_factor,
            grid_n_rows = 1,
            grid_n_cols = 1,
            namespace = namespace,
            name = name,
            parent = parent, 
            verbose = verbose,
            add_settings_tab = add_settings_tab,
            )

    def _init_shared_data(self):
        
        is_server = False
        
        self.shared_data_clients.append(RhcStatus(is_server=is_server,
                                            namespace=self.namespace, 
                                            verbose=True, 
                                            vlevel=VLevel.V2))
        
        self.shared_data_clients[0].run()

    def _post_shared_init(self):
        
        self.grid_n_rows = 7 + int(self.shared_data_clients[0].n_contacts/2)

        self.grid_n_cols = 2

    def _initialize(self):
        
        cluster_size = self.shared_data_clients[0].cluster_size
        cluster_idx_legend = [""] * cluster_size
        for i in range(cluster_size):
            cluster_idx_legend[i] = str(i)
        
        self.rt_plotters.append(RtPlotWindow(data_dim=1,
                                    n_data = 1,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Controllers counter", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=["counter"], 
                                    ylabel="[int]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=cluster_size,
                                    n_data = 1,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Registration flags", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=cluster_idx_legend, 
                                    ylabel="[bool]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=cluster_size,
                                    n_data = 1,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Fails counter", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=cluster_idx_legend, 
                                    ylabel="[int]"))

        self.rt_plotters.append(RtPlotWindow(data_dim=cluster_size,
                                    n_data = 1,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Failure flags", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=cluster_idx_legend, 
                                    ylabel="[bool]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=cluster_size,
                                    n_data = 1,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Fail. index", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=cluster_idx_legend, 
                                    ylabel="[float]"))

        self.rt_plotters.append(RtPlotWindow(data_dim=cluster_size,
                                    n_data = 1,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Reset flags", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=cluster_idx_legend, 
                                    ylabel="[bool]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=cluster_size,
                                    n_data = 1,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Trigger flags", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=cluster_idx_legend, 
                                    ylabel="[bool]"))

        self.rt_plotters.append(RtPlotWindow(data_dim=cluster_size,
                                    n_data = 1,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Activation flags", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=cluster_idx_legend, 
                                    ylabel="[bool]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=cluster_size,
                                    n_data = 1,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Rhc Cost", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=cluster_idx_legend, 
                                    ylabel="[float]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=cluster_size,
                                    n_data = 1,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Rhc Constraint Violation", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=cluster_idx_legend, 
                                    ylabel="[float]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=cluster_size,
                                    n_data = 1,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Rhc Iteration Number", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=cluster_idx_legend, 
                                    ylabel="[int]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=cluster_size,
                                    n_data = self.shared_data_clients[0].n_nodes,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Rhc Cost on nodes", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=cluster_idx_legend, 
                                    ylabel="[float]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=cluster_size,
                                    n_data = self.shared_data_clients[0].n_nodes,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Rhc Constraint Violation on nodes", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=cluster_idx_legend, 
                                    ylabel="[float]"))
        
        for i in range(self.shared_data_clients[0].n_contacts):
            self.rt_plotters.append(RtPlotWindow(data_dim=cluster_size,
                                        n_data = self.shared_data_clients[0].n_nodes,
                                        update_data_dt=self.update_data_dt, 
                                        update_plot_dt=self.update_plot_dt,
                                        window_duration=self.window_duration, 
                                        parent=None, 
                                        base_name=f"Rhc Step Flag - contact n {i}", 
                                        window_buffer_factor=self.window_buffer_factor, 
                                        legend_list=cluster_idx_legend, 
                                        ylabel="[float]"))
        
        self.grid.addFrame(self.rt_plotters[0].base_frame, 0, 0)
        self.grid.addFrame(self.rt_plotters[1].base_frame, 0, 1)
        self.grid.addFrame(self.rt_plotters[2].base_frame, 1, 0)
        self.grid.addFrame(self.rt_plotters[3].base_frame, 1, 1)
        self.grid.addFrame(self.rt_plotters[4].base_frame, 2, 0)
        self.grid.addFrame(self.rt_plotters[5].base_frame, 2, 1)
        self.grid.addFrame(self.rt_plotters[6].base_frame, 3, 0)
        self.grid.addFrame(self.rt_plotters[7].base_frame, 3, 1)
        self.grid.addFrame(self.rt_plotters[8].base_frame, 4, 0)
        self.grid.addFrame(self.rt_plotters[9].base_frame, 4, 1)
        self.grid.addFrame(self.rt_plotters[10].base_frame, 5, 0)
        self.grid.addFrame(self.rt_plotters[11].base_frame, 5, 1)
        self.grid.addFrame(self.rt_plotters[12].base_frame, 6, 0)

        n_rows = int(self.shared_data_clients[0].n_contacts/2)
        for i in range(n_rows):
            for j in range(2):
                self.grid.addFrame(self.rt_plotters[12 + 2 * i + j].base_frame, 6 + i, j)

    def _finalize_grid(self):
        
        widget_utils = WidgetUtils()

        settings_frames = []

        node_index_slider = widget_utils.generate_complex_slider(
                        parent=None, 
                        parent_layout=None,
                        min_shown=f"{0}", min= 0, 
                        max_shown=f"{self.shared_data_clients[0].n_nodes - 1}", 
                        max=self.shared_data_clients[0].n_nodes - 1, 
                        init_val_shown=f"{0}", init=0, 
                        title="node index slider", 
                        callback=self._update_node_idx)
        
        settings_frames.append(node_index_slider)
        
        self.grid.addToSettings(settings_frames)

    def _update_node_idx(self,
                    idx: int):

        self.current_node_index = idx

        self.grid.settings_widget_list[0].current_val.setText(f'{idx}')
        
        # only cost and constr on nodes
        self.rt_plotters[11].rt_plot_widget.switch_to_data(data_idx = self.current_node_index)
        self.rt_plotters[12].rt_plot_widget.switch_to_data(data_idx = self.current_node_index)
        n_contacts = self.shared_data_clients[0].n_contacts
        for i in range(n_contacts):
            self.rt_plotters[13+i].rt_plot_widget.switch_to_data(data_idx = self.current_node_index)

    def update(self,
            index: int):

        # index not used here (no dependency on cluster index)

        if not self._terminated:
            
            # read data on shared memory
            self.shared_data_clients[0].controllers_counter.synch_all(read = True, 
                                                        retry=False)
            self.shared_data_clients[0].registration.synch_all(read = True, 
                                                        retry=False)
            self.shared_data_clients[0].controllers_fail_counter.synch_all(read = True, 
                                                        retry=False)
            self.shared_data_clients[0].fails.synch_all(read = True, 
                                                        retry=False)
            self.shared_data_clients[0].resets.synch_all(read = True, 
                                                        retry=False)
            self.shared_data_clients[0].trigger.synch_all(read = True, 
                                                        retry=False)
            self.shared_data_clients[0].activation_state.synch_all(read = True, 
                                                        retry=False)
            self.shared_data_clients[0].rhc_cost.synch_all(read = True, 
                                                    retry=False)
            self.shared_data_clients[0].rhc_constr_viol.synch_all(read = True, 
                                                    retry=False)
            self.shared_data_clients[0].rhc_n_iter.synch_all(read = True, 
                                                    retry=False)
            self.shared_data_clients[0].rhc_nodes_cost.synch_all(read = True, 
                                                    retry=False)
            self.shared_data_clients[0].rhc_nodes_constr_viol.synch_all(read = True, 
                                                    retry=False)
            self.shared_data_clients[0].rhc_step_var.synch_all(read = True, 
                                                    retry=False)
            self.shared_data_clients[0].rhc_fail_idx.synch_all(read = True, 
                                                    retry=False)
            
            self.rt_plotters[0].rt_plot_widget.update(self.shared_data_clients[0].controllers_counter.get_numpy_mirror())
            self.rt_plotters[1].rt_plot_widget.update(self.shared_data_clients[0].registration.get_numpy_mirror())
            self.rt_plotters[2].rt_plot_widget.update(self.shared_data_clients[0].controllers_fail_counter.get_numpy_mirror())
            self.rt_plotters[3].rt_plot_widget.update(self.shared_data_clients[0].fails.get_numpy_mirror())
            self.rt_plotters[4].rt_plot_widget.update(self.shared_data_clients[0].rhc_fail_idx.get_numpy_mirror())
            self.rt_plotters[5].rt_plot_widget.update(self.shared_data_clients[0].resets.get_numpy_mirror())
            self.rt_plotters[6].rt_plot_widget.update(self.shared_data_clients[0].trigger.get_numpy_mirror())
            self.rt_plotters[7].rt_plot_widget.update(self.shared_data_clients[0].activation_state.get_numpy_mirror())
            self.rt_plotters[8].rt_plot_widget.update(self.shared_data_clients[0].rhc_cost.get_numpy_mirror())
            self.rt_plotters[9].rt_plot_widget.update(self.shared_data_clients[0].rhc_constr_viol.get_numpy_mirror())
            self.rt_plotters[10].rt_plot_widget.update(self.shared_data_clients[0].rhc_n_iter.get_numpy_mirror())
            self.rt_plotters[11].rt_plot_widget.update(self.shared_data_clients[0].rhc_nodes_cost.get_numpy_mirror())
            self.rt_plotters[12].rt_plot_widget.update(self.shared_data_clients[0].rhc_nodes_constr_viol.get_numpy_mirror())

            n_contacts = self.shared_data_clients[0].n_contacts
            tot_data = self.shared_data_clients[0].rhc_step_var.get_numpy_mirror()
            for i in range(n_contacts):
                start_idx = self.shared_data_clients[0].n_nodes * i
                single_contact_data = tot_data[:, start_idx:(start_idx+self.shared_data_clients[0].n_nodes)]
                self.rt_plotters[12 + i].rt_plot_widget.update(single_contact_data)
            
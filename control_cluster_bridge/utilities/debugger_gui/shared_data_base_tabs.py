from PyQt5.QtWidgets import QWidget

from control_cluster_bridge.utilities.debugger_gui.gui_exts import SharedDataWindow
from control_cluster_bridge.utilities.debugger_gui.plot_utils import RtPlotWindow

from control_cluster_bridge.utilities.rhc_defs import RhcTaskRefs, RobotCmds, RobotState, ContactState, RHCInternal
from control_cluster_bridge.utilities.control_cluster_defs import RHCStatus

from SharsorIPCpp.PySharsorIPC import VLevel
from control_cluster_bridge.utilities.shared_mem import SharedMemClient, SharedStringArray

import torch

from control_cluster_bridge.utilities.defs import jnt_names_client_name
from control_cluster_bridge.utilities.defs import additional_data_name
from control_cluster_bridge.utilities.defs import n_contacts_name

from control_cluster_bridge.utilities.debugger_gui.plot_utils import WidgetUtils

class RhcTaskRefWindow(SharedDataWindow):

    def __init__(self, 
            update_data_dt: int,
            update_plot_dt: int,
            window_duration: int,
            window_buffer_factor: int = 2,
            namespace = "",
            parent: QWidget = None, 
            verbose = False):

        self.n_contacts = -1
        self.cluster_size = -1
    
        super().__init__(update_data_dt = update_data_dt,
            update_plot_dt = update_plot_dt,
            window_duration = window_duration,
            window_buffer_factor = window_buffer_factor,
            grid_n_rows = 2,
            grid_n_cols = 3,
            namespace = namespace,
            name = "RhcTaskRefWindow",
            parent = parent, 
            verbose = verbose)
    
    def _initialize(self):

        self.rt_plotters.append(RtPlotWindow(n_data=self.n_contacts, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Contact flags", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=None, 
                    ylabel=""))
        
        self.rt_plotters.append(RtPlotWindow(n_data=1, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Task mode", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=["task mode code"]))
        
        self.rt_plotters.append(RtPlotWindow(n_data=7, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Base pose", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=["p_x", "p_y", "p_z", 
                                "q_w", "q_i", "q_j", "q_k"]))
        
        self.rt_plotters.append(RtPlotWindow(n_data=7, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="CoM pose", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=["p_x", "p_y", "p_z", 
                                "q_w", "q_i", "q_j", "q_k"]))
        
        self.rt_plotters.append(RtPlotWindow(n_data=10, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Phase params", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=["duration", 
                                "p0_x", "p0_y", "p0_z",
                                "p1_x", "p1_y", "p1_z", 
                                "c",
                                "d0", "d1"]
                    ))
        
        self.grid.addFrame(self.rt_plotters[0].base_frame, 0, 0)
        self.grid.addFrame(self.rt_plotters[1].base_frame, 0, 1)
        self.grid.addFrame(self.rt_plotters[2].base_frame, 1, 0)
        self.grid.addFrame(self.rt_plotters[3].base_frame, 1, 1)
        self.grid.addFrame(self.rt_plotters[4].base_frame, 1, 2)

    def _init_shared_data(self):
        
        self.shared_data_clients.append(SharedMemClient(name=n_contacts_name(), 
                                    namespace=self.namespace, 
                                    dtype=torch.int64, 
                                    verbose=self.verbose))
        
        self.shared_data_clients[0].attach()

        self.n_contacts = self.shared_data_clients[0].tensor_view[0, 0].item()

        self.shared_data_clients.append(RhcTaskRefs( 
                                            n_contacts=self.n_contacts,
                                            index=0,
                                            q_remapping=None,
                                            namespace=self.namespace,
                                            dtype=torch.float32, 
                                            verbose=self.verbose)
                                        )
        
        self.cluster_size = \
                    self.shared_data_clients[1].shared_memman.n_rows
        
        # now we know how big the cluster is

        # view of remaining RhcTaskRefs
        for i in range(1, self.cluster_size):

            self.shared_data_clients.append(RhcTaskRefs( 
                n_contacts=self.n_contacts,
                index=i,
                q_remapping=None,
                namespace=self.namespace,
                dtype=torch.float32, 
                verbose=self.verbose))
    
    def _post_shared_init(self):

        pass

    def update(self):

        if not self._terminated:
            
            self.rt_plotters[0].rt_plot_widget.update(self.shared_data_clients[self.cluster_idx + 1].phase_id.get_contacts().numpy())
            self.rt_plotters[1].rt_plot_widget.update(self.shared_data_clients[self.cluster_idx + 1].phase_id.phase_id.numpy())
            self.rt_plotters[2].rt_plot_widget.update(self.shared_data_clients[self.cluster_idx + 1].base_pose.get_pose().numpy())
            self.rt_plotters[3].rt_plot_widget.update(self.shared_data_clients[self.cluster_idx + 1].com_pose.get_com_pose().numpy())
            self.rt_plotters[4].rt_plot_widget.update(self.shared_data_clients[self.cluster_idx + 1].phase_id.get_flight_param().numpy())

class RhcCmdsWindow(SharedDataWindow):

    def __init__(self, 
            update_data_dt: int,
            update_plot_dt: int,
            window_duration: int,
            window_buffer_factor: int = 2,
            namespace = "",
            parent: QWidget = None, 
            verbose = False):

        self.jnt_number = -1 
        self.jnt_names = []
        self.add_data_length = -1

        super().__init__(update_data_dt = update_data_dt,
            update_plot_dt = update_plot_dt,
            window_duration = window_duration,
            window_buffer_factor = window_buffer_factor,
            grid_n_rows = 2,
            grid_n_cols = 3,
            namespace = namespace,
            name = "RhcCmdsWindow",
            parent = parent, 
            verbose = verbose)

    def _initialize(self):

        self.rt_plotters.append(RtPlotWindow(n_data=self.jnt_number, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="RHC command q", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=self.jnt_names, 
                    ylabel="[rad]"))
        
        self.rt_plotters.append(RtPlotWindow(n_data=self.jnt_number, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="RHC command v", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=self.jnt_names, 
                    ylabel="[rad/s]"))
        
        self.rt_plotters.append(RtPlotWindow(n_data=self.jnt_number, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt, 
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="RHC command effort", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=self.jnt_names, 
                    ylabel="[Nm]"))
        
        self.rt_plotters.append(RtPlotWindow(n_data=self.add_data_length, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="additional info", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=None))
        
        self.grid.addFrame(self.rt_plotters[0].base_frame, 0, 0)
        self.grid.addFrame(self.rt_plotters[1].base_frame, 0, 1)
        self.grid.addFrame(self.rt_plotters[2].base_frame, 1, 0)
        self.grid.addFrame(self.rt_plotters[3].base_frame, 1, 1)

    def _init_shared_data(self):
        
        self.shared_data_clients.append(SharedStringArray(length=-1, 
                                    name=jnt_names_client_name(), 
                                    namespace=self.namespace, 
                                    is_server=False, 
                                    verbose=self.verbose))
        self.shared_data_clients[0].start()

        self.jnt_names = self.shared_data_clients[0].read()

        self.shared_data_clients.append(SharedMemClient(name=additional_data_name(), 
                                    namespace=self.namespace, 
                                    dtype=torch.int64, 
                                    verbose=self.verbose))
        self.shared_data_clients[1].attach()
        self.add_data_length = self.shared_data_clients[1].tensor_view[0, 0].item()

        self.shared_data_clients.append(RobotCmds(n_dofs=len(self.jnt_names), 
                                    index=0, 
                                    jnt_remapping=None, # we see everything as seen on the simulator side 
                                    add_info_size=self.add_data_length, 
                                    dtype=torch.float32, 
                                    namespace=self.namespace,
                                    verbose=self.verbose))
        
        self.cluster_size = \
            self.shared_data_clients[2].shared_memman.n_rows
        
        self.jnt_number = self.shared_data_clients[2].n_dofs
        # now we know how big the cluster is

        # view of remaining RobotCmds
        for i in range(1, self.cluster_size):

            self.shared_data_clients.append(RobotCmds(n_dofs=len(self.jnt_names), 
                                                index=i, 
                                                jnt_remapping=None, # we see everything as seen on the simulator side 
                                                add_info_size=self.add_data_length, 
                                                dtype=torch.float32, 
                                                namespace=self.namespace,
                                                verbose=self.verbose)
                                            )
    
    def _post_shared_init(self):

        pass

    def update(self):
        
        if not self._terminated:
            
            self.rt_plotters[0].rt_plot_widget.update(self.shared_data_clients[self.cluster_idx + 2].jnt_cmd.q.numpy())
            self.rt_plotters[1].rt_plot_widget.update(self.shared_data_clients[self.cluster_idx + 2].jnt_cmd.v.numpy())
            self.rt_plotters[2].rt_plot_widget.update(self.shared_data_clients[self.cluster_idx + 2].jnt_cmd.eff.numpy())
            self.rt_plotters[3].rt_plot_widget.update(self.shared_data_clients[self.cluster_idx + 2].slvr_state.info.numpy())

class RhcStateWindow(SharedDataWindow):

    def __init__(self, 
            update_data_dt: int,
            update_plot_dt: int,
            window_duration: int,
            window_buffer_factor: int = 2,
            namespace = "",
            parent: QWidget = None, 
            verbose = False):
        
        self.cluster_size = -1 

        super().__init__(update_data_dt = update_data_dt,
            update_plot_dt = update_plot_dt,
            window_duration = window_duration,
            grid_n_rows = 2,
            grid_n_cols = 3,
            window_buffer_factor = window_buffer_factor,
            namespace = namespace,
            name = "RhcStateWindow",
            parent = parent, 
            verbose = verbose)

    def _initialize(self):

        self.rt_plotters.append(RtPlotWindow(n_data=self.shared_data_clients[1].root_state.p.shape[1], 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Root position", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=["p_x", "p_y", "p_z"], 
                    ylabel="[m]"))
        
        self.rt_plotters.append(RtPlotWindow(n_data=self.shared_data_clients[1].root_state.q.shape[1], 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Root orientation", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=["q_w", "q_i", "q_j", "q_k"]))
        
        self.rt_plotters.append(RtPlotWindow(n_data=self.shared_data_clients[1].root_state.v.shape[1], 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt, 
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Base linear vel.", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=["v_x", "v_y", "v_z"], 
                    ylabel="[m/s]"))
        
        self.rt_plotters.append(RtPlotWindow(n_data=self.shared_data_clients[1].root_state.omega.shape[1], 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt, 
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Base angular vel.",
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=["omega_x", "omega_y", "omega_z"], 
                    ylabel="[rad/s]"))
        
        self.rt_plotters.append(RtPlotWindow(n_data=self.shared_data_clients[1].jnt_state.n_dofs, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Joints q",
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=self.jnt_names, 
                    ylabel="[rad]"))
        
        self.rt_plotters.append(RtPlotWindow(n_data=self.shared_data_clients[1].jnt_state.n_dofs, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Joints v",
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=self.jnt_names, 
                    ylabel="[rad/s]"))
        
        # root state
        self.grid.addFrame(self.rt_plotters[0].base_frame, 0, 0)
        self.grid.addFrame(self.rt_plotters[1].base_frame, 0, 1)
        self.grid.addFrame(self.rt_plotters[2].base_frame, 1, 0)
        self.grid.addFrame(self.rt_plotters[3].base_frame, 1, 1)
        
        # joint state
        self.grid.addFrame(self.rt_plotters[4].base_frame, 0, 2)
        self.grid.addFrame(self.rt_plotters[5].base_frame, 1, 2)

    def _init_shared_data(self):
        
        self.shared_data_clients.append(SharedStringArray(length=-1, 
                                    name=jnt_names_client_name(), 
                                    namespace=self.namespace, 
                                    is_server=False, 
                                    verbose=self.verbose))
        self.shared_data_clients[0].start()

        self.jnt_names = self.shared_data_clients[0].read()

        self.shared_data_clients.append(RobotState(n_dofs=len(self.jnt_names), 
                                    index=0, 
                                    jnt_remapping=None, 
                                    q_remapping=None, 
                                    namespace=self.namespace,
                                    dtype=torch.float32, 
                                    verbose=self.verbose))
        self.cluster_size = self.shared_data_clients[1].shared_memman.n_rows

        # now we know how big the cluster is

        # view of remaining RobotState
        for i in range(1, self.cluster_size):

            self.shared_data_clients.append(RobotState(n_dofs=len(self.jnt_names), 
                                    index=i, 
                                    jnt_remapping=None, 
                                    q_remapping=None, 
                                    namespace=self.namespace,
                                    dtype=torch.float32, 
                                    verbose=self.verbose))
            
    def _post_shared_init(self):

        pass

    def update(self):

        if not self._terminated:
            
            # root state
            self.rt_plotters[0].rt_plot_widget.update(self.shared_data_clients[self.cluster_idx + 1].root_state.p.numpy())
            self.rt_plotters[1].rt_plot_widget.update(self.shared_data_clients[self.cluster_idx + 1].root_state.q.numpy())
            self.rt_plotters[2].rt_plot_widget.update(self.shared_data_clients[self.cluster_idx + 1].root_state.v.numpy())
            self.rt_plotters[3].rt_plot_widget.update(self.shared_data_clients[self.cluster_idx + 1].root_state.omega.numpy())

            # joint state
            self.rt_plotters[4].rt_plot_widget.update(self.shared_data_clients[self.cluster_idx + 1].jnt_state.q.numpy())
            self.rt_plotters[5].rt_plot_widget.update(self.shared_data_clients[self.cluster_idx + 1].jnt_state.v.numpy())

class RhcContactStatesWindow(SharedDataWindow):

    def __init__(self, 
            update_data_dt: int,
            update_plot_dt: int,
            window_duration: int,
            window_buffer_factor: int = 2,
            namespace = "",
            parent: QWidget = None, 
            verbose = False):
        
        self.n_sensors = -1
        self.contact_info_size = -1
        self.contact_names = []

        super().__init__(update_data_dt = update_data_dt,
            update_plot_dt = update_plot_dt,
            window_duration = window_duration,
            window_buffer_factor = window_buffer_factor,
            grid_n_rows = 2,
            grid_n_cols = 3,
            namespace = namespace,
            name = "RhcContactStatesWindow",
            parent = parent, 
            verbose = verbose)

    def _init_shared_data(self):
        
        self.shared_data_clients.append(ContactState(index=0, 
                                    namespace=self.namespace,
                                    dtype=torch.float32, 
                                    verbose=self.verbose))
        self.cluster_size = self.shared_data_clients[0].shared_memman.n_rows

        # view of rhc references
        for i in range(1, self.cluster_size):

            self.shared_data_clients.append(ContactState(index=i, 
                                namespace=self.namespace,
                                dtype=torch.float32, 
                                verbose=self.verbose))
    
    def _post_shared_init(self):

        self.contact_names = self.shared_data_clients[0].contact_names

        self.n_sensors = self.shared_data_clients[0].n_contacts

        self.contact_info_size = round(self.shared_data_clients[0].shared_memman.n_cols / self.n_sensors)

        if self.n_sensors <= 0:

            warning = "[{self.__class__.__name__}]" + f"[{self.journal.warning}]" \
                + f": terminating since no contact sensor was found."
            
            print(warning)

            self.terminate()

        import math

        grid_size = math.ceil(math.sqrt(self.n_sensors))

        # distributing plots over a square grid
        self.grid_n_rows = grid_size
        self.grid_n_cols = grid_size

    def _initialize(self):

        # distribute plots on each row
        counter = 0
        for i in range(0, self.grid_n_rows):
            
            for j in range(0, self.grid_n_cols):
            
                if (counter < self.n_sensors):
                    
                    self.rt_plotters.append(RtPlotWindow(n_data=self.contact_info_size, 
                                update_data_dt=self.update_data_dt, 
                                update_plot_dt=self.update_plot_dt,
                                window_duration=self.window_duration, 
                                parent=None, 
                                base_name=f"Net contact force on link {self.contact_names[counter]}", 
                                window_buffer_factor=self.window_buffer_factor, 
                                legend_list=["f_x", "f_y", "f_z"], 
                                ylabel="[N]")
                                )

                    self.grid.addFrame(self.rt_plotters[counter].base_frame, i, j)

                    counter = counter + 1

    def update(self,
            index: int = 0):

        if not self._terminated:
                        
            for i in range(0, self.n_sensors):
                
                # cluster index is updated manually
                self.rt_plotters[i].rt_plot_widget.update(self.shared_data_clients[self.cluster_idx].contact_state.get(self.contact_names[i]).numpy())

class RhcInternalData(SharedDataWindow):

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
        settings_title = "SETTINGS (RhcInternalData)"
        ):
        
        self.add_settings_tab = add_settings_tab

        self.is_cost = is_cost
        self.is_constraint = is_constraint

        self.cluster_size = -1

        self.current_node_index = 0

        self.names = []
        self.dims = []
        self.n_nodes = -1

        self.name = name
        
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
        
        self.rhc_status_info = RHCStatus(is_server=is_server,
                            namespace=self.namespace, 
                            verbose=self.verbose,
                            vlevel=VLevel.V1)

        self.rhc_status_info.run()
        
        self.cluster_size = self.rhc_status_info.trigger.n_rows

        self.rhc_status_info.close() # we don't need the client anymore

        enable_costs = False
        enable_constr = False

        if self.is_cost:
            
            enable_costs = True

        if self.is_constraint:
            
            enable_constr = True

        config = RHCInternal.Config(is_server=is_server, 
                        enable_q=False, 
                        enable_v=False, 
                        enable_a=False, 
                        enable_a_dot=False, 
                        enable_f=False,
                        enable_f_dot=False, 
                        enable_eff=False,
                        enable_costs=enable_costs, 
                        enable_constr=enable_constr)

        # view of rhc internal data
        for i in range(0, self.cluster_size):

            self.shared_data_clients.append(RHCInternal(config=config,
                                            namespace=self.namespace,
                                            rhc_index = i,
                                            is_server=is_server,
                                            verbose=self.verbose,
                                            vlevel=VLevel.V1))
        
        # run clients
        for i in range(0, self.cluster_size):

            self.shared_data_clients[i].run()

    def _post_shared_init(self):
        
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

    def _initialize(self):
        
        base_name = ""

        if self.is_cost:

            base_name = "Cost name"

        if self.is_constraint:

            base_name = "Constr.name"

        # distribute plots on each row
        counter = 0
        for i in range(0, self.grid_n_rows):
            
            for j in range(0, self.grid_n_cols):
            
                if (counter < len(self.names)):
                    
                    legend_list = [""] * self.dims[counter]
                    for k in range(self.dims[counter]):
                        
                        legend_list = str(k)

                    self.rt_plotters.append(RtPlotWindow(n_data=self.dims[counter], 
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

    def update(self,
            index: int):

        if not self._terminated:
            
            for i in range(0, len(self.names)):
                
                # iterate over data names (i.e. plots)

                # self.rt_plotters[i].rt_plot_widget.update(self.shared_data_clients[index].)
                a = 1

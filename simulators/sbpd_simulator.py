import numpy as np
from obstacles.sbpd_map import SBPDMap
from simulators.simulator import Simulator
from trajectory.trajectory import Trajectory, SystemConfig


class SBPDSimulator(Simulator):
    name = 'SBPD_Simulator'

    def __init__(self, params):
        assert(params.obstacle_map_params.obstacle_map is SBPDMap)
        super(SBPDSimulator, self).__init__(params=params)

    def get_observation(self, config=None, pos_n3=None, **kwargs):
        """
        Return the robot's observation from configuration config
        or pos_nk3.
        """
        return self.obstacle_map.get_observation(config=config, pos_n3=pos_n3, **kwargs)

    def get_observation_from_data_dict_and_model(self, data_dict, model):
        """
        Returns the robot's observation from the data inside data_dict,
        using parameters specified by the model.
        """
        if hasattr(model, 'occupancy_grid_positions_ego_1mk12'):
            kwargs = {'occupancy_grid_positions_ego_1mk12':
                      model.occupancy_grid_positions_ego_1mk12}
        else:
            kwargs = {}

        img_nmkd = self.get_observation(pos_n3=data_dict['vehicle_state_nk3'][:, 0],
                                        **kwargs)
        return img_nmkd

    def generate_cost(self, data_dict):
        '''
        Given the optimal waypoint, get the expert's cost of that waypoint from
        the MPC problem.
        '''
        cost = []  
        # First state in 30-step episode
        # The 30-step episode is a trajectory snippet in which the agent
        # tries to reach the waypoint
        current_states = data_dict['vehicle_state_nk3'][:, 0]  
        waypoints = data_dict['optimal_waypoint_n3']   
        goal_positions = data_dict['goal_position_n2']
        for i in range(len(goal_positions)):
            current_pos = current_states[i, :2]
            current_heading = current_states[i, 2]
            goal_pos = goal_positions[i].reshape(1, 2)
            waypt_pos = waypoints[i, :2]
            waypt_heading = waypoints[i, 2]
            # Get the FMM map with this goal position
            fmm_map = self._init_fmm_map(goal_pos)
            self._update_obj_fn(fmm_map)

            n = 1  # Batch size
            k = 1  # 1-step "trajectories" -- just waypoints
            start_config = SystemConfig(dt=0, n=n, k=k, 
                                        position_nk2=current_pos.reshape((n, k, 2)), 
                                        heading_nk1=current_heading.reshape((n, k, 1)))
            goal_config = SystemConfig(dt=0, n=n, k=k, 
                                        position_nk2=waypt_pos.reshape((n, k, 2)),
                                        heading_nk1=waypt_heading.reshape((n, k, 1)))
            # Take spline trajectory from current robot state to waypoint
            # and take the cost of that
            c, _ = self.planner.eval_objective(start_config, goal_config)  # Tensor
            c = c[0].numpy()
            cost.append(c)  

        return np.array(cost)

    def _reset_obstacle_map(self, rng):
        """
        For SBPD the obstacle map does not change
        between episodes.
        """
        return False

    def _update_fmm_map(self):
        """
        For SBPD the obstacle map does not change,
        so just update the goal position.
        """
        if hasattr(self, 'fmm_map'):
            goal_pos_n2 = self.goal_config.position_nk2()[:, 0]
            self.fmm_map.change_goal(goal_pos_n2)
        else:
            self.fmm_map = self._init_fmm_map()
        self._update_obj_fn()

    def _init_obstacle_map(self, rng):
        """ Initializes the sbpd map."""
        p = self.params.obstacle_map_params
        return p.obstacle_map(p)

    def _render_obstacle_map(self, ax):
        p = self.params
        self.obstacle_map.render_with_obstacle_margins(ax, start_config=self.start_config,
                                                       margin0=p.avoid_obstacle_objective.obstacle_margin0,
                                                       margin1=p.avoid_obstacle_objective.obstacle_margin1)

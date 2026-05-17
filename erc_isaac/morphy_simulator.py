"""
MorphySimulator: Clean interface for Morphy soft drone simulation.

This class provides direct control over the Morphy drone simulation without
high-level configuration abstractions. It allows direct application of:
- Forces and torques to motor bodies
- Torques to soft joints
"""
from typing import Optional
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, PhysxCfg
from isaaclab.utils import configclass
from erc_isaac.airframe_encoding import AIRFRAME_CONSTANTS


@configclass
class MorphySceneCfg(InteractiveSceneCfg):
    """Configuration for Morphy scene."""
    dome_light = AssetBaseCfg(prim_path='/World/Light', spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    robot: ArticulationCfg = ArticulationCfg(prim_path='/World/envs/env_.*/Robot', spawn=sim_utils.UsdFileCfg(usd_path='', articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=None)), init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)), actuators={'soft_joints': ImplicitActuatorCfg(joint_names_expr=['arm_segment_1_to_arm_segment_2_1', 'arm_segment_1_to_arm_segment_2_3'], stiffness=0.0, damping=0.0, effort_limit_sim=20000.0, velocity_limit_sim=10000.0, effort_limit=20000.0, velocity_limit=10000.0, friction=0.0, dynamic_friction=0.0, viscous_friction=0.0, armature=0.0)})

class MorphySimulator:
    _shared_sim_context = None

    def __init__(self, usd_path: str, n_envs: int=1, dt: float=0.001, device: str='cuda:0', gravity: tuple[float, float, float]=(0.0, 0.0, -9.81), motor_model: str='motors_disabled', motor_directions: list[int]=None, revolute_joint_damping_coeff: float=None, env_spacing: float=2.0, fix_base: bool=False, results_dir: str | None=None):
        self.usd_path = usd_path
        self.n_envs = n_envs
        self.dt = dt
        self.device = device
        self.gravity = gravity
        self.motor_model = motor_model
        self.motor_directions_list = AIRFRAME_CONSTANTS['motor_directions']
        self.damping = revolute_joint_damping_coeff
        self.env_spacing = env_spacing
        self.fix_base = fix_base
        self.results_dir = Path(results_dir) if results_dir is not None else Path('results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.track_joint_dynamics = True
        self._joint_track = None
        self._setup_simulation()
        self._detect_rigid_arms()
        if self.has_soft_joints:
            assert revolute_joint_damping_coeff is not None, 'revolute_joint_damping_coeff must be provided for soft airframes'
        self._load_robot()
        self._load_drone_parameters()
        if self.has_soft_joints:
            self._build_joint_torque_functions()
        self._compute_robot_mass()
        print(f'[MorphySimulator] Initialized with {n_envs} environments, dt={dt}s, gravity={gravity}, motor_model={motor_model}, has_soft_joints={self.has_soft_joints}')

    def _setup_simulation(self):
        if MorphySimulator._shared_sim_context is None:
            sim_cfg = sim_utils.SimulationCfg(dt=self.dt, device=self.device, gravity=self.gravity, physx=PhysxCfg(enable_stabilization=False))
            MorphySimulator._shared_sim_context = SimulationContext(sim_cfg)
            MorphySimulator._shared_sim_context.set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.5])
            print('[MorphySimulator] Created new shared SimulationContext')
        else:
            print('[MorphySimulator] Reusing existing shared SimulationContext')
        self.sim = MorphySimulator._shared_sim_context

    def _detect_rigid_arms(self):
        """Detect whether the USD has rigid arms by reading the morphy:rigid_arms attribute."""
        from pxr import Usd
        stage = Usd.Stage.Open(self.usd_path)
        root_prim = stage.GetPrimAtPath('/quadrotor')
        if not root_prim.IsValid():
            raise RuntimeError(f"Root prim '/quadrotor' not found in {self.usd_path}")
        rigid_attr = root_prim.GetAttribute('morphy:rigid_arms')
        if not rigid_attr.IsValid() or rigid_attr.Get() is None:
            raise RuntimeError(f'Failed to load morphy:rigid_arms from {self.usd_path}.')
        is_rigid = rigid_attr.Get() is True
        self.has_soft_joints = not is_rigid
        print(f'[MorphySimulator] Detected rigid_arms={is_rigid}, has_soft_joints={self.has_soft_joints}')

    def _load_robot(self):
        print(f'[MorphySimulator] num_envs: {self.n_envs}, env_spacing: {self.env_spacing}')
        scene_cfg = MorphySceneCfg(num_envs=self.n_envs, env_spacing=self.env_spacing, replicate_physics=False)
        scene_cfg.robot.spawn.usd_path = self.usd_path
        if not self.has_soft_joints:
            scene_cfg.robot.actuators = {}
            scene_cfg.robot.init_state.joint_pos = {}
            scene_cfg.robot.init_state.joint_vel = {}
        if self.fix_base:
            scene_cfg.robot.spawn.articulation_props.fix_root_link = True
            print('[MorphySimulator] Fixing base using ArticulationRootPropertiesCfg.fix_root_link')
        self.scene = InteractiveScene(scene_cfg)
        self.robot = self.scene['robot']
        self.sim.reset()
        if self.has_soft_joints:
            soft_joint_names = ['arm_segment_1_to_arm_segment_2_1', 'arm_segment_1_to_arm_segment_2_3']
            self.soft_joint_indices = [self.robot.joint_names.index(name) for name in soft_joint_names]
            print(f'[MorphySimulator] Soft joint indices: {self.soft_joint_indices}')
        else:
            self.soft_joint_indices = []
            print('[MorphySimulator] No soft joints (rigid airframe)')
        print(f'[MorphySimulator] Loaded {self.n_envs} robot(s) from {self.usd_path}')
        if self.fix_base:
            print(f'[MorphySimulator] Base is fixed: {self.robot.is_fixed_base}')

    def _load_drone_parameters(self):
        """Load drone-specific torque parameters from USD custom attributes."""
        if not self.has_soft_joints:
            self.torque_params = {}
            print('[MorphySimulator] Rigid airframe - skipping torque parameter loading')
            return
        from pxr import Usd
        stage = Usd.Stage.Open(self.usd_path)
        root_prim = stage.GetPrimAtPath('/quadrotor')
        if not root_prim.IsValid():
            raise RuntimeError(f"Root prim '/quadrotor' not found in {self.usd_path}")
        param_names = ['torque_fn_left', 'torque_fn_right']
        self.torque_params = {}
        for name in param_names:
            attr = root_prim.GetAttribute(f'morphy:{name}')
            if not attr.IsValid() or attr.Get() is None:
                raise RuntimeError(f'Failed to load morphy:{name} from {self.usd_path}.')
            self.torque_params[name] = str(attr.Get())
        print(f'[MorphySimulator] Loaded drone torque parameters from {self.usd_path}:')
        for (name, value) in self.torque_params.items():
            print(f'  {name}: {value}')

    def _build_joint_torque_functions(self):
        """Build separate torque functions for left and right arms from USD strings."""
        p = self.torque_params
        fn_str_left = p['torque_fn_left']
        fn_str_right = p['torque_fn_right']
        print('[MorphySimulator] Evaluating torque functions:')
        print(f'  Left:  {fn_str_left}')
        print(f'  Right: {fn_str_right}')
        self.torque_fn_left = self._execute_torque_function_string(fn_str_left, 'left')
        self.torque_fn_right = self._execute_torque_function_string(fn_str_right, 'right')
        angles_left = self._diagnostic_torque_plot_angles(self.torque_fn_left, 'left')
        angles_right = self._diagnostic_torque_plot_angles(self.torque_fn_right, 'right')
        with torch.no_grad():
            torques_left = self.torque_fn_left(angles_left)
            torques_right = self.torque_fn_right(angles_right)
        (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(angles_left.cpu().numpy() * 180 / torch.pi, torques_left.cpu().numpy(), 'b-', linewidth=2)
        ax1.axvline(x=0.0, color='k', linestyle='--', alpha=0.3)
        ax1.axhline(y=0.0, color='k', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Joint Angle (degrees)')
        ax1.set_ylabel('Torque (N*m)')
        ax1.set_title('Left Arm Torque Response')
        ax1.grid(True, alpha=0.3)
        ax2.plot(angles_right.cpu().numpy() * 180 / torch.pi, torques_right.cpu().numpy(), 'r-', linewidth=2)
        ax2.axvline(x=0.0, color='k', linestyle='--', alpha=0.3)
        ax2.axhline(y=0.0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Joint Angle (degrees)')
        ax2.set_ylabel('Torque (N*m)')
        ax2.set_title('Right Arm Torque Response')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'torque_response.jpeg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f'[MorphySimulator] Built separate left/right torque functions, damping={self.damping}')

    def _diagnostic_torque_plot_angles(self, torque_fn, side: str) -> torch.Tensor:
        if hasattr(torque_fn, 'erc_angle_min_rad') and hasattr(torque_fn, 'erc_angle_max_rad'):
            angle_min = float(torque_fn.erc_angle_min_rad)
            angle_max = float(torque_fn.erc_angle_max_rad)
        elif side == 'left':
            angle_min = -20.0 * float(torch.pi) / 180.0
            angle_max = 120.0 * float(torch.pi) / 180.0
        elif side == 'right':
            angle_min = -120.0 * float(torch.pi) / 180.0
            angle_max = 20.0 * float(torch.pi) / 180.0
        else:
            raise ValueError(f'Unexpected soft-joint side: {side}')
        if angle_min >= angle_max:
            raise ValueError(f'Invalid diagnostic torque plot domain for {side}: {angle_min} >= {angle_max}')
        return torch.linspace(angle_min, angle_max, 1000, device=self.device)

    def _execute_torque_function_string(self, fn_str: str, side: str):
        namespace = {'torch': torch}
        exec(fn_str, namespace)
        fn_name = f'torque_fn_{side}'
        if fn_name not in namespace:
            raise RuntimeError(f'Torque function string did not define {fn_name}')
        torque_fn = namespace[fn_name]
        if not callable(torque_fn):
            raise RuntimeError(f'{fn_name} is not callable')
        return torque_fn

    def step(self, extra_joint_torques: Optional[torch.Tensor]=None):
        """Advance one simulation step.

        Args:
            extra_joint_torques: Optional ``(n_envs, n_soft_joints)`` tensor of
                additional torques to add on top of the ERC profile torques.
        """
        if self.has_soft_joints:
            (joint_pos, joint_vel) = self.get_joint_state()
            (interp, damp) = self.compute_joint_torques_decomposed(joint_pos, joint_vel)
            if self.track_joint_dynamics:
                self._record_joint_data(joint_pos, joint_vel, interp, damp)
            torques = interp + damp
            if extra_joint_torques is not None:
                torques = torques + extra_joint_torques
            env_indices = torch.arange(self.n_envs, dtype=torch.long, device=self.device)
            self.robot.set_joint_effort_target(torques, joint_ids=self.soft_joint_indices, env_ids=env_indices)
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.dt)

    def reset(self, env_ids: Optional[torch.Tensor]=None):
        if env_ids is None:
            env_ids = torch.arange(self.n_envs, device=self.device)
        self.robot.reset(env_ids)
        print(f'[MorphySimulator] Reset {len(env_ids)} environment(s)')

    def get_base_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get base (root) state.

        Returns:
            position: (n_envs, 3) in world frame
            orientation: (n_envs, 4) quaternion (w, x, y, z) in world frame
            linear_vel: (n_envs, 3) in world frame
            angular_vel: (n_envs, 3) in world frame
        """
        return (self.robot.data.root_pos_w, self.robot.data.root_quat_w, self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w)

    def get_joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get soft joint state (2 joints only).

        Returns:
            joint_pos: (n_envs, 2) soft joint positions in radians, or (n_envs, 0) if rigid
            joint_vel: (n_envs, 2) soft joint velocities in rad/s, or (n_envs, 0) if rigid
        """
        if not self.has_soft_joints:
            return (torch.zeros(self.n_envs, 0, device=self.device), torch.zeros(self.n_envs, 0, device=self.device))
        return (self.robot.data.joint_pos[:, self.soft_joint_indices], self.robot.data.joint_vel[:, self.soft_joint_indices])

    def get_body_state(self, body_names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Get state of specific bodies.

        Args:
            body_names: List of body names to query

        Returns:
            positions: (n_envs, n_bodies, 3) positions in world frame
            orientations: (n_envs, n_bodies, 4) quaternions (w, x, y, z) in world frame
        """
        body_indices = [self.robot.body_names.index(name) for name in body_names]
        positions = self.robot.data.body_pos_w[:, body_indices, :]
        orientations = self.robot.data.body_quat_w[:, body_indices, :]
        return (positions, orientations)

    def compute_joint_torques(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
        """Compute soft joint torques using separate left/right functions and damping.

        Args:
            joint_pos: (n_envs, 2) soft joint positions in radians [left, right]
            joint_vel: (n_envs, 2) soft joint velocities in rad/s

        Returns:
            torques: (n_envs, 2) computed joint torques
        """
        (interpolated, damping) = self.compute_joint_torques_decomposed(joint_pos, joint_vel)
        return interpolated + damping

    def compute_joint_torques_decomposed(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute soft joint torques, returning interpolated and damping components separately.

        Args:
            joint_pos: (n_envs, 2) soft joint positions in radians [left, right]
            joint_vel: (n_envs, 2) soft joint velocities in rad/s

        Returns:
            interpolated_torque: (n_envs, 2) torque from runtime function (position-based)
            damping_torque: (n_envs, 2) torque from damping (velocity-based)
        """
        left_torque = self.torque_fn_left(joint_pos[:, 0:1])
        right_torque = self.torque_fn_right(joint_pos[:, 1:2])
        interpolated_torque = torch.cat([left_torque, right_torque], dim=1)
        damping_torque = -self.damping * joint_vel
        return (interpolated_torque, damping_torque)

    def _compute_robot_mass(self):
        """Compute total robot mass from URDF."""
        body_masses = self.robot.root_physx_view.get_masses()
        self.robot_mass = body_masses[0].sum().item()
        print(f'[MorphySimulator] Robot mass: {self.robot_mass:.3f} kg')

    def _record_joint_data(self, joint_pos, joint_vel, interp, damp):
        """Record one timestep of joint data for env[0]."""
        if self._joint_track is None:
            self._joint_track = {'time': [], 'pos': [], 'vel': [], 'interp': [], 'damp': []}
            self._joint_track_step = 0
        self._joint_track['time'].append(self._joint_track_step * self.dt)
        self._joint_track['pos'].append(joint_pos[0].cpu().numpy().copy())
        self._joint_track['vel'].append(joint_vel[0].cpu().numpy().copy())
        self._joint_track['interp'].append(interp[0].cpu().numpy().copy())
        self._joint_track['damp'].append(damp[0].cpu().numpy().copy())
        self._joint_track_step += 1

    def plot_joint_dynamics(self):
        """Plot recorded joint dynamics to PDF (one page per arm)."""
        import numpy as np
        from matplotlib.backends.backend_pdf import PdfPages

        def fmt(v):
            return str(v).replace('.', '_').replace('-', 'neg')
        output_path = self.results_dir / f'joint_dynamics_dt_{fmt(self.dt)}_damping_{fmt(self.damping)}.pdf'
        self.track_joint_dynamics = False
        data = self._joint_track
        self._joint_track = None
        if data is None or len(data['time']) == 0:
            print('[MorphySimulator] No joint data recorded, skipping plot')
            return
        t = np.array(data['time'])
        pos = np.degrees(np.array(data['pos']))
        vel = np.degrees(np.array(data['vel']))
        interp = np.array(data['interp'])
        damp = np.array(data['damp'])
        with PdfPages(output_path) as pdf:
            for (arm_idx, arm_name) in enumerate(['Left', 'Right']):
                (fig, axes) = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
                axes[0].plot(t, pos[:, arm_idx], 'b-', lw=1.5)
                axes[0].set_ylabel('Position (deg)')
                axes[0].axhline(0, color='k', ls='--', alpha=0.3)
                axes[0].grid(True, alpha=0.3)
                axes[1].plot(t, pos[:, arm_idx], 'b-', lw=1.5)
                axes[1].set_ylabel('Position (deg)\n[zoomed]')
                axes[1].set_ylim(-5, 5)
                axes[1].axhline(0, color='k', ls='--', alpha=0.3)
                axes[1].grid(True, alpha=0.3)
                axes[2].plot(t, vel[:, arm_idx], 'g-', lw=1.5)
                axes[2].set_ylabel('Velocity (deg/s)')
                axes[2].axhline(0, color='k', ls='--', alpha=0.3)
                axes[2].grid(True, alpha=0.3)
                axes[3].plot(t, interp[:, arm_idx], 'r-', lw=1.5)
                axes[3].set_ylabel('Interpolated (Nm)')
                axes[3].axhline(0, color='k', ls='--', alpha=0.3)
                axes[3].grid(True, alpha=0.3)
                axes[4].plot(t, damp[:, arm_idx], 'm-', lw=1.5)
                axes[4].set_ylabel('Damping (Nm)')
                axes[4].set_xlabel('Time (s)')
                axes[4].axhline(0, color='k', ls='--', alpha=0.3)
                axes[4].grid(True, alpha=0.3)
                fig.suptitle(f'{arm_name} Arm Joint Dynamics', fontsize=14)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        print(f'[MorphySimulator] Saved joint dynamics to {output_path}')

    def cleanup(self):
        """Clean up simulation resources."""
        self.scene.reset()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()
        self._joint_track_step += 1

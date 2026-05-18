"""Module for numerical operations in Arm KDL."""

# Standard imports
from dataclasses import dataclass

# Third-party imports
import numpy as np
import casadi

# Local imports
from .arm_kdl import ArmKdl
from .utils import timed


class WeightedMovingFilter:
    """
    A weighted moving average filter for smoothing trajectory data.
    """

    def __init__(self, weights, dim):
        """
        Initialize the weighted moving filter.

        Args:
            weights (np.array): Array of weights for the moving average
            dim (int): Dimension of the data to be filtered
        """
        self.weights = weights
        self.window_size = len(weights)
        self.buffer = np.zeros((self.window_size, dim))
        self.current_index = 0
        self.filled = False

    def add_data(self, data):
        """
        Add new data to the filter buffer.

        Args:
            data (np.array): New data point to add
        """
        self.buffer[self.current_index] = data
        self.current_index = (self.current_index + 1) % self.window_size
        if self.current_index == 0:
            self.filled = True

    @property
    def filtered_data(self):
        """
        Calculate the weighted average of the data in the buffer.

        Returns:
            np.array: Filtered data
        """
        if self.filled:
            # All weights used
            return np.sum(self.buffer * self.weights[:, np.newaxis], axis=0)
        else:
            # Use only the filled portion of the buffer
            used_weights = self.weights[: self.current_index]
            normalized_weights = used_weights / np.sum(used_weights)
            return np.sum(
                self.buffer[: self.current_index] * normalized_weights[:, np.newaxis],
                axis=0,
            )


class ArmKdlNumerical(ArmKdl):
    """Extended Arm kinematics and dynamics class with numerical IK solver"""

    def __init__(self, arm_type: str = "play_short", eef_type: str = "G2"):
        """
        Initialize the ArmKdlNumerical class.

        Args:
            arm_type (str, optional): Type of arm. Defaults to "play_short".
            eef_type (str, optional): Type of end effector. Defaults to "G2".
        """
        # Initialize parent class
        super().__init__(arm_type, eef_type)

        # Initialize casadi models for numerical IK
        self._initialize_numerical_ik()

        # Initialize data for trajectory smoothing
        self.init_data = np.zeros(6)  # Initial joint angles
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 6)

        # Warm-up: Perform a dummy solve to absorb initialization overhead
        default_pose = self.forward_kinematics(np.zeros(6))  # Use zero pose for warm-up
        self.solve_numerical_ik(default_pose, current_q=np.zeros(6))

    def _initialize_numerical_ik(self):
        """Initialize CasADi optimization problem for numerical IK"""
        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", 6, 1)  # Joint angles
        self.cTf = casadi.SX.sym("tf", 4, 4)  # Target pose

        # Forward kinematics as a CasADi function
        def fk_casadi(q):
            """Forward kinematics function for CasADi"""
            # Individual transform matrices
            T = casadi.SX.eye(4)
            for i in range(6):
                # Build transformation matrix for each joint
                cosq = casadi.cos(q[i] + self.dh.theta_offset[i])
                sinq = casadi.sin(q[i] + self.dh.theta_offset[i])
                cosa = casadi.cos(self.dh.alpha[i])
                sina = casadi.sin(self.dh.alpha[i])

                T_i = casadi.SX.zeros(4, 4)
                T_i[0, 0] = cosq
                T_i[0, 1] = -sinq
                T_i[0, 2] = 0
                T_i[0, 3] = self.dh.a[i]
                T_i[1, 0] = sinq * cosa
                T_i[1, 1] = cosq * cosa
                T_i[1, 2] = -sina
                T_i[1, 3] = -self.dh.d[i] * sina
                T_i[2, 0] = sinq * sina
                T_i[2, 1] = cosq * sina
                T_i[2, 2] = cosa
                T_i[2, 3] = self.dh.d[i] * cosa
                T_i[3, 0] = 0
                T_i[3, 1] = 0
                T_i[3, 2] = 0
                T_i[3, 3] = 1

                T = T @ T_i

            # Add final link transform
            T_7 = casadi.SX.zeros(4, 4)
            T_7[0, 0] = 1
            T_7[0, 1] = 0
            T_7[0, 2] = 0
            T_7[0, 3] = 0
            T_7[1, 0] = 0
            T_7[1, 1] = 1
            T_7[1, 2] = 0
            T_7[1, 3] = 0
            T_7[2, 0] = 0
            T_7[2, 1] = 0
            T_7[2, 2] = 1
            T_7[2, 3] = self.dh.d[6]
            T_7[3, 0] = 0
            T_7[3, 1] = 0
            T_7[3, 2] = 0
            T_7[3, 3] = 1

            T = T @ T_7 @ self.dh.end_convert_matrix

            return T

        # Create CasADi function for FK
        self.fk_func = casadi.Function("fk", [self.cq], [fk_casadi(self.cq)])

        # Compute pose error
        pose = self.fk_func(self.cq)
        pos_error = pose[:3, 3] - self.cTf[:3, 3]

        # Compute orientation error using logarithm of rotation matrix difference
        R_current = pose[:3, :3]
        R_target = self.cTf[:3, :3]
        R_error = R_current @ R_target.T

        # Orientation error computation
        # We need to extract the axis-angle representation from the rotation matrix
        tr = R_error[0, 0] + R_error[1, 1] + R_error[2, 2]
        theta = casadi.acos((tr - 1) / 2)

        # Compute the orientation error vector
        # For small rotations or when near identity
        rot_error_x = (R_error[2, 1] - R_error[1, 2]) / 2
        rot_error_y = (R_error[0, 2] - R_error[2, 0]) / 2
        rot_error_z = (R_error[1, 0] - R_error[0, 1]) / 2
        rot_error = casadi.vertcat(rot_error_x, rot_error_y, rot_error_z)

        # Define error functions
        self.translational_error = casadi.Function(
            "translational_error", [self.cq, self.cTf], [pos_error]
        )

        self.rotational_error = casadi.Function(
            "rotational_error", [self.cq, self.cTf], [rot_error]
        )

        # Define optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(6)  # Joint variables
        self.var_q_last = self.opti.parameter(6)  # Previous joint values for smoothing
        self.param_tf = self.opti.parameter(4, 4)  # Target pose

        # Cost function components
        self.translational_cost = casadi.sumsqr(
            self.translational_error(self.var_q, self.param_tf)
        )
        self.rotation_cost = casadi.sumsqr(
            self.rotational_error(self.var_q, self.param_tf)
        )
        self.regularization_cost = casadi.sumsqr(
            self.var_q
        )  # Prefer configurations near zero
        self.smooth_cost = casadi.sumsqr(
            self.var_q - self.var_q_last
        )  # Prefer smooth transitions

        # Joint limits
        lower_limits = np.array([limit[0] for limit in self.dh.joints_limit])
        upper_limits = np.array([limit[1] for limit in self.dh.joints_limit])

        # Set optimization constraints and objective
        self.opti.subject_to(self.opti.bounded(lower_limits, self.var_q, upper_limits))

        # Weight the different costs
        self.opti.minimize(
            50 * self.translational_cost
            + self.rotation_cost
            + 0.02 * self.regularization_cost
            + 0.1 * self.smooth_cost
        )

        # Adjust Ipopt options for faster convergence
        opts = {
            "ipopt": {
                "print_level": 0,
                "max_iter": 50,
                "tol": 1e-5,
            },  # Reduced max_iter and relaxed tol
            "print_time": False,
            "calc_lam_p": False,
        }
        self.opti.solver("ipopt", opts)

    # @timed
    def solve_numerical_ik(self, target_pose, current_q=None):
        """
        Solve IK numerically using optimization.

        Args:
            target_pose (np.array): 4x4 target pose matrix
            current_q (np.array, optional): Current joint angles. Defaults to None.

        Returns:
            tuple: (joint_angles, joint_torques)
                - joint_angles: Solution joint angles
                - joint_torques: Computed joint torques
        """
        # Set initial guess
        if current_q is not None:
            self.init_data = current_q

        self.opti.set_initial(self.var_q, self.init_data)

        # Set parameters
        self.opti.set_value(self.param_tf, target_pose)
        self.opti.set_value(self.var_q_last, self.init_data)

        try:
            # Solve optimization problem
            sol = self.opti.solve()

            # Get solution
            sol_q = self.opti.value(self.var_q)

            # Apply smoothing filter
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            # Update initial data for next iteration
            self.init_data = sol_q

            return sol_q

        except Exception as e:
            print(f"ERROR in numerical IK convergence: {e}")

            # Try to get the debug value
            try:
                sol_q = self.opti.debug.value(self.var_q)
                self.smooth_filter.add_data(sol_q)
                sol_q = self.smooth_filter.filtered_data
                self.init_data = sol_q
                return sol_q
            except:
                # If optimization completely fails, return the current configuration
                if current_q is not None:
                    return current_q, np.zeros(6)
                else:
                    return self.init_data

    def solve_ik(self, target_pose, method="analytical", ref_pos=None, current_q=None):
        """
        Unified interface for solving IK with multiple methods.

        Args:
            target_pose (np.array): 4x4 target pose matrix
            method (str, optional): IK method - "analytical" or "numerical". Defaults to "analytical".
            ref_pos (np.array, optional): Reference joint position for analytical IK. Defaults to None.
            current_q (np.array, optional): Current joint position for numerical IK. Defaults to None.

        Returns:
            tuple: (joint_angles, joint_torques)
                - joint_angles: Solution joint angles
                - joint_torques: Joint torques (zeros for analytical)
        """
        if method == "analytical":
            solutions = super().inverse_kinematics(target_pose, ref_pos)
            if solutions is None or len(solutions) == 0:
                print("Analytical IK found no solutions, trying numerical IK")
                return self.solve_numerical_ik(target_pose, current_q)

            # Take the first solution from analytical IK
            q = solutions[0]
            # Calculate torques
            # v = np.zeros(6)  # Zero velocity
            # a = np.zeros(6)  # Zero acceleration
            # tau = self.inverse_dynamics(q, v, a)
            # return q, tau
            return q

        elif method == "numerical":
            return self.solve_numerical_ik(target_pose, current_q)

        else:
            raise ValueError(
                f"Unknown IK method: {method}. Use 'analytical' or 'numerical'."
            )


def main():
    """
    Example usage of the ArmKdlNumerical class.
    """
    # Create an instance of the extended class
    arm_kdl = ArmKdlNumerical(eef_type="G2")

    # Test forward kinematics
    print("== Testing Forward Kinematics ==")
    test_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end_pose = arm_kdl.forward_kinematics(test_q)
    print("End effector pose from zero position:")
    print(end_pose)

    # Test analytical IK
    print("\n== Testing Analytical IK ==")
    joints = arm_kdl.inverse_kinematics(end_pose, ref_pos=test_q)
    if joints:
        print("Analytical IK solution:")
        print(joints[0])
        fk_result = arm_kdl.forward_kinematics(joints[0])
        print("FK from solution:")
        print(fk_result)
    else:
        print("No analytical IK solution found")

    # Test numerical IK
    print("\n== Testing Numerical IK ==")
    q_numerical = arm_kdl.solve_ik(end_pose, method="numerical", current_q=test_q)
    print("Numerical IK solution:")
    print(q_numerical)
    fk_numerical = arm_kdl.forward_kinematics(q_numerical)
    print("FK from numerical solution:")
    print(fk_numerical)

    # Test unified interface
    print("\n== Testing Unified IK Interface ==")
    q_unified = arm_kdl.solve_ik(end_pose, method="analytical", ref_pos=test_q)
    print("Unified IK solution:")
    print(q_unified)

    # Test with slightly modified pose
    print("\n== Testing with Modified Pose ==")
    modified_pose = end_pose.copy()
    modified_pose[0, 3] += 0.05  # Move 5cm in x direction
    q_modified = arm_kdl.solve_ik(modified_pose, method="numerical", current_q=test_q)
    print("Numerical IK solution for modified pose:")
    print(q_modified)
    fk_modified = arm_kdl.forward_kinematics(q_modified)
    print("FK from modified solution:")
    print(fk_modified)


if __name__ == "__main__":
    main()

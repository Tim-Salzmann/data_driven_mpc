import rosbag
import rospy
import pandas as pd

import numpy as np

from src.experiments.point_tracking_and_record import make_record_dict
from src.quad_mpc.create_ros_gp_mpc import ROSGPMPC
from src.utils.utils import jsonify


def odometry_parse(odom_msg):
    p = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z]
    q = [odom_msg.pose.pose.orientation.w, odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
         odom_msg.pose.pose.orientation.z]
    v = [odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y, odom_msg.twist.twist.linear.z]
    w = [odom_msg.twist.twist.angular.x, odom_msg.twist.twist.angular.y, odom_msg.twist.twist.angular.z]

    return np.array(p + q + v + w)


def thrust_dynamics_model2(motor_tau, thrust, thrust_des, dt):
    if motor_tau < 1e-12:
        return thrust_des
    tau_inv = 1 / motor_tau
    thrust_out = (
            tau_inv ** 2 * (thrust_des - 2 * (thrust * thrust_des) ** 0.5 + thrust) * dt ** 2
            + 2 * tau_inv * ((thrust * thrust_des) ** 0.5 - thrust) * dt
            + thrust
    )
    return thrust_out


def thrust_dynamics_model(motor_tau, thrust, thrust_des, dt):
    tau_inv = 1 / motor_tau
    thrust_out = (
            (thrust_des - 2 * (thrust * thrust_des) ** 0.5 + thrust) * np.exp(-2 * dt * tau_inv)
            + 2 * (thrust - (thrust * thrust_des) ** 0.5) * np.exp(-dt * tau_inv)
            + thrust
    )
    return thrust_out


def system_identification(quad_mpc, bag, ctrl_frequency):
    thrust = None
    control = {'t': [], 'thrust': []}
    for topic, msg, t in bag.read_messages(topics=['control']):
        t = msg.header.stamp
        desired_thrust = np.array(msg.thrusts)
        if thrust is None:
            thrust = desired_thrust
        t = t + rospy.Duration.from_sec(quad_mpc.quad.comm_delay) + rospy.Duration.from_sec(0.001)
        for dt in np.arange(0.001, 0.021, step=0.001):
            thrust = thrust_dynamics_model2(quad_mpc.quad.motor_tau, thrust, desired_thrust, 0.001)
            t_at = t + rospy.Duration.from_sec(dt) #if quad_mpc.quad.motor_tau > 0. else rospy.Duration.from_sec(0.001))
            control['t'].append(t_at.to_sec())
            control['thrust'].append(thrust)

    control = pd.DataFrame(control)
    control = control.set_index('t')
    control = control[~control.index.duplicated(keep='last')]

    rec_dict = make_record_dict(state_dim=13)
    recording = False
    last_state_msg = None
    for topic, msg, t in bag.read_messages(topics=['recording_ctrl', 'state']):
        if topic == 'recording_ctrl':
            recording = msg.data
            if not recording:
                last_state_msg = None
        if topic == 'state' and recording:
            if last_state_msg is not None:
                last_state_idx = control.index.get_loc(last_state_msg.header.stamp.to_sec(), method='nearest')
                curr_state_idx = control.index.get_loc(msg.header.stamp.to_sec(), method='nearest')
                u0 = control['thrust'].iloc[last_state_idx] / quad_mpc.quad.max_thrust
                u1 = control['thrust'].iloc[curr_state_idx] / quad_mpc.quad.max_thrust
                if np.all(np.abs(u0 - u1) < 0.01):
                    x_0 = odometry_parse(last_state_msg)
                    x_f = odometry_parse(msg)
                    u = np.vstack([u0, u1]).mean(0)
                    dt = msg.header.stamp.to_sec() - last_state_msg.header.stamp.to_sec()
                    x_pred, _ = quad_mpc.quad_mpc.forward_prop(x_0, u, t_horizon=dt, use_gp=False)
                    x_pred = x_pred[-1, np.newaxis, :]

                    rec_dict['state_in'] = np.append(rec_dict['state_in'], x_0[np.newaxis, :], axis=0)
                    rec_dict['input_in'] = np.append(rec_dict['input_in'], u[np.newaxis, :], axis=0)
                    rec_dict['state_out'] = np.append(rec_dict['state_out'], x_f[np.newaxis, :], axis=0)
                    rec_dict['state_ref'] = np.append(rec_dict['state_ref'], np.zeros_like(x_f[np.newaxis, :]), axis=0)
                    rec_dict['timestamp'] = np.append(rec_dict['timestamp'], msg.header.stamp.to_sec())
                    rec_dict['dt'] = np.append(rec_dict['dt'], dt)
                    rec_dict['state_pred'] = np.append(rec_dict['state_pred'], x_pred, axis=0)
                    rec_dict['error'] = np.append(rec_dict['error'], x_f - x_pred, axis=0)

            last_state_msg = msg

    # Save datasets
    for key in rec_dict.keys():
        print(key, " ", rec_dict[key].shape)
        rec_dict[key] = jsonify(rec_dict[key])
    df = pd.DataFrame(rec_dict)
    df.to_csv('/Users/TimSalzmann/Documents/Study/PhD/Code/Neural-MPC/experiments/data_driven_mpc/ros_gp_mpc/data/agisim_dataset/train/dataset_001.csv', index=True, header=True)


if __name__ == '__main__':
    control_frequency = 50
    ros_quad_mpc = ROSGPMPC(1, 10, 1/control_frequency, 'kingfisher')

    bag = rosbag.Bag('/Users/TimSalzmann/Documents/Study/PhD/Code/Neural-MPC/experiments/data_driven_mpc/ros_gp_mpc/data/agisim_dataset/train/dataset_001.bag')
    system_identification(ros_quad_mpc, bag, control_frequency)
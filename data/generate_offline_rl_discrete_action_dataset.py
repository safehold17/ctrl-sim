import json
import os
import pickle
import time
import hydra
import numpy as np
import nocturne
import torch
import torch.nn.functional as F
import imageio

from nocturne import Simulation
from nocturne.bicycle_model import BicycleModel
from cfgs.config import set_display_window
from utils.data import get_object_type_str, get_road_type_str
from utils.geometry import angle_sub 
from utils.sim import get_sim, get_ground_truth_states, get_road_data, compute_reward
from utils.k_disks_helpers import (
    transform_box_corners_from_vocab,
    get_local_state_transition,
    transform_box_corners_from_local_state,
    get_global_next_state
)
from tqdm import tqdm

POS_X_IDX = 0
POS_Y_IDX = 1
VEL_X_IDX = 2
VEL_Y_IDX = 3
HEAD_IDX = 4
LEN_IDX = 5
WID_IDX = 6
EXIST_IDX = -1


def load_k_disks_vocab(cfg):
    vocab_path = getattr(cfg, "k_disks_vocab_path", None)
    if vocab_path is None:
        vocab_path = os.path.join(cfg.project_root, "K-disks actions", "k_disks_vocab_384_10Hz_seed42.pkl")
    elif not os.path.isabs(vocab_path):
        vocab_path = os.path.join(cfg.project_root, vocab_path)
    with open(vocab_path, "rb") as f:
        vocab = np.array(pickle.load(f)["V"])
    return vocab


def build_agent_states_for_k_disks(gt_data_dict, vehicles):
    vehicle_ids = []
    agent_states = []
    for veh in vehicles:
        veh_id = veh.getID()
        if veh_id not in gt_data_dict:
            continue
        traj = np.array(gt_data_dict[veh_id]["traj"])
        pos = traj[:, :2]
        heading = traj[:, 2]
        speed = traj[:, 3]
        existence = traj[:, 4]
        length = traj[:, 7]
        width = veh.getWidth()

        vel_x = speed * np.cos(heading)
        vel_y = speed * np.sin(heading)
        width_col = np.ones_like(speed) * width

        agent_state = np.stack(
            [pos[:, 0], pos[:, 1], vel_x, vel_y, heading, length, width_col, existence],
            axis=1,
        )
        agent_states.append(agent_state)
        vehicle_ids.append(veh_id)

    if len(agent_states) == 0:
        return np.zeros((0, 0, 0)), []

    return np.stack(agent_states), vehicle_ids


def rollout_k_disks(agent_states, vocab, cfg, delta_t):
    """Compute discrete k-disk actions from continuous trajectories."""
    num_agents = agent_states.shape[0]  # agent第0维度的大小
    num_steps = agent_states.shape[1] - 1

    states = np.zeros_like(agent_states)
    actions = np.zeros((num_agents, num_steps))

    states[:, 0] = agent_states[:, 0]
    tokenize_with_nucleus_sampling = getattr(cfg, "tokenize_with_nucleus_sampling", False)
    tokenization_temperature = getattr(cfg, "tokenization_temperature", 1.0)
    tokenization_nucleus = getattr(cfg, "tokenization_nucleus", 0.9)

    for t in range(num_steps):
        valid_timestep = np.logical_and(  # 逻辑与函数
            agent_states[:, t, EXIST_IDX],
            agent_states[:, t + 1, EXIST_IDX],
        )
        states[:, t, EXIST_IDX] = valid_timestep.astype(int)

        corner_0_x = -1 * states[:, t, LEN_IDX] / 2
        corner_0_y = -1 * states[:, t, WID_IDX] / 2
        corner_1_x = -1 * states[:, t, LEN_IDX] / 2
        corner_1_y = states[:, t, WID_IDX] / 2
        corner_2_x = states[:, t, LEN_IDX] / 2
        corner_2_y = states[:, t, WID_IDX] / 2
        corner_3_x = states[:, t, LEN_IDX] / 2
        corner_3_y = -1 * states[:, t, WID_IDX] / 2

        box_corners = np.array(
            [
                [corner_0_x, corner_0_y],
                [corner_1_x, corner_1_y],
                [corner_2_x, corner_2_y],
                [corner_3_x, corner_3_y],
            ]
        ).transpose(2, 0, 1)

        box_corners_vocab = transform_box_corners_from_vocab(box_corners, vocab)

        current_state = states[:, t, [POS_X_IDX, POS_Y_IDX, HEAD_IDX]]
        gt_next_state = agent_states[:, t + 1, [POS_X_IDX, POS_Y_IDX, HEAD_IDX]]

        local_state_transitions = get_local_state_transition(
            current_state=current_state, next_state=gt_next_state
        )

        box_corners_local_state = transform_box_corners_from_local_state(
            box_corners, local_state_transitions
        )

        err = np.linalg.norm(box_corners_vocab - box_corners_local_state[:, None, :, :], axis=-1).mean(2)

        if tokenize_with_nucleus_sampling:
            err_torch = torch.from_numpy(-err)
            action_probs = F.softmax(err_torch / tokenization_temperature, dim=1)
            sorted_probs, sorted_indices = torch.sort(action_probs, dim=-1, descending=True)

            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            selected_actions = cum_probs < tokenization_nucleus
            selected_actions[:, 0] = True

            next_action_dis = torch.zeros_like(err_torch)
            next_action_dis.scatter_(1, sorted_indices, selected_actions * sorted_probs)
            next_action_dis = next_action_dis / next_action_dis.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            next_actions = torch.multinomial(next_action_dis, 1)[:, 0]
            next_actions = next_actions.numpy()
        else:
            next_actions = np.argmin(err, axis=1)

        next_actions[~valid_timestep] = 0  # 将无效时间步的动作设为0
        actions[:, t] = next_actions

        next_state_pos_heading = get_global_next_state(current_state, vocab[next_actions])

        next_v = (next_state_pos_heading[:, :2] - current_state[:, :2]) / delta_t
        next_exists = np.zeros(num_agents).astype(int)
        next_state = np.array(
            [
                next_state_pos_heading[:, 0],
                next_state_pos_heading[:, 1],
                next_v[:, 0],
                next_v[:, 1],
                next_state_pos_heading[:, 2],
                states[:, t, LEN_IDX],
                states[:, t, WID_IDX],
                next_exists,
            ]
        ).transpose(1, 0)

        next_state[~valid_timestep] = 0
        states[:, t + 1] = next_state

    return states, actions


def collect_data(cfg, dt, steps, output_path, files_path, files, chunk, k_disks_vocab):
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # loop through all training files
    for file in tqdm(chunk):
        gt_data_dict = get_ground_truth_states(cfg, files_path, files, file, dt, steps)
        sim = get_sim(cfg, files_path, files, file)
        scenario = sim.getScenario()
        vehicles = scenario.vehicles()

        agent_states, k_disk_vehicle_ids = build_agent_states_for_k_disks(gt_data_dict, vehicles)
        k_disk_actions = None
        veh_id_to_idx = {veh_id: idx for idx, veh_id in enumerate(k_disk_vehicle_ids)}
        if agent_states.size > 0:
            _, k_disk_actions = rollout_k_disks(agent_states, k_disks_vocab, cfg, dt)

        for veh in vehicles:
            if veh.getID() in gt_data_dict.keys():
                veh.expert_control = False
                veh.physics_simulated = True
            else:
                veh.expert_control = True
                veh.physics_simulated = False
        
        # Collect vehicle data
        vehicle_data_dict = {}
        goal_dict = {}
        goal_dist_normalizer = {}

        for t in range(steps):
            for veh in vehicles:
                veh_id = veh.getID()
                
                if veh_id not in gt_data_dict.keys():
                    continue

                if t == 0:
                    goal_pos = np.array([veh.target_position.x, veh.target_position.y])
                    goal_heading = veh.target_heading
                    goal_speed = veh.target_speed
                    gt_traj_data = np.array(gt_data_dict[veh_id]['traj'])
                    idx_disappear = np.where(gt_traj_data[:, 4] == 0)[0]
                    if len(idx_disappear) > 0:
                        idx_disappear = idx_disappear[0] - 1
                        if np.linalg.norm(gt_traj_data[idx_disappear, :2] - goal_pos) > 0.0:
                            goal_pos = gt_traj_data[idx_disappear, :2]
                            goal_heading = gt_traj_data[idx_disappear, 2]
                            goal_speed = gt_traj_data[idx_disappear, 3]
                    
                    vehicle_data_dict[veh_id] = {
                        "position": [], # [{'x': float, 'y': float}, ...]
                        "velocity": [],  # [{'x': float, 'y': float}, ...]
                        "heading": [], # [float, ...]
                        "existence": [],
                        "acceleration": [], # [float, ...]
                        "steering": [], # [float, ...]
                        "reward": [], # [float, ...]
                        "goal_position": {'x': goal_pos[0], 'y': goal_pos[1]},
                        "goal_heading": goal_heading,
                        "goal_speed": goal_speed,
                        "width": veh.getWidth(),
                        "length": veh.getLength(),
                        "type": get_object_type_str(veh)
                    }

                    goal_dict[veh_id] = {
                            'pos': goal_pos,
                            'heading': goal_heading,
                            'speed': goal_speed
                        }

                    # Precompute goal-dist normalizer (used for reward computation)
                    obj_pos = veh.getPosition()
                    obj_pos = np.array([obj_pos.x, obj_pos.y])
                    dist = np.linalg.norm(obj_pos - goal_pos)
                    goal_dist_normalizer[veh_id] = dist
                
                # action is only defined if state at next timestep is defined
                veh_exists = gt_data_dict[veh_id]['traj'][t][4] and gt_data_dict[veh_id]['traj'][t+1][4]
                # once we encounter the first missing timestep, all future timesteps are also missing
                # this is because we need contiguous sequence to push through nocturne simulator
                if t > 0 and vehicle_data_dict[veh_id]["existence"][-1] == 0:
                    veh_exists = 0
                
                if not veh_exists:
                    acceleration = 0.0
                    steering = 0.0
                    veh.setPosition(-1000000, -1000000)  # make cars disappear if they are out of actions
                else:
                    bike_model = BicycleModel(x=gt_data_dict[veh_id]['traj'][t+1][0],
                                                y=gt_data_dict[veh_id]['traj'][t+1][1],
                                                theta=gt_data_dict[veh_id]['traj'][t+1][2],
                                                vel=gt_data_dict[veh_id]['traj'][t+1][3],
                                                L=gt_data_dict[veh_id]['traj'][t+1][-1],
                                                dt=0.1)
                    
                    accel, steer, _, _ = bike_model.backward(prev_pos=np.array([veh.getPosition().x,veh.getPosition().y]), 
                                                                prev_theta=veh.getHeading(),
                                                                prev_vel=veh.getSpeed())
                    veh_action = [accel, steer]

                    acceleration = veh_action[0]
                    steering = veh_action[1]

                if acceleration > 0.0:
                    veh.acceleration = acceleration
                else:
                    veh.brake(np.abs(acceleration))
                veh.steering = steering

                # Compute reward 
                reward = compute_reward(cfg.nocturne['rew_cfg'], veh, goal_dict[veh_id], goal_dist_normalizer[veh_id], vehicle_data_dict, collision_fix=cfg.nocturne.collision_fix)

                # Append vehicle state data
                vehicle_data_dict[veh_id]["position"].append({'x': veh.getPosition().x, 'y': veh.getPosition().y})
                vehicle_data_dict[veh_id]["velocity"].append({'x': veh.velocity().x, 'y': veh.velocity().y})
                vehicle_data_dict[veh_id]["heading"].append(veh.getHeading())
                vehicle_data_dict[veh_id]["existence"].append(veh_exists)
                vehicle_data_dict[veh_id]["acceleration"].append(acceleration)
                vehicle_data_dict[veh_id]["steering"].append(steering)
                vehicle_data_dict[veh_id]["reward"].append(reward)

            sim.step(dt)

        if k_disk_actions is not None:
            for veh_id, idx in veh_id_to_idx.items():
                if veh_id in vehicle_data_dict:
                    vehicle_data_dict[veh_id]["k_action"] = [int(a) for a in k_disk_actions[idx]]

        road_data = get_road_data(scenario)

        # Save data to files
        file_name = f"{files[file].split('.')[0]}_physics.json"
        export_data = {"name": file_name, "objects": [*vehicle_data_dict.values()], "roads": road_data}

        with open(os.path.join(output_path, file_name), 'w') as file:
            json.dump(export_data, file)

        sim.reset()


@hydra.main(version_base=None, config_path="../cfgs/", config_name="config")
def main(cfg):
    
    if cfg.offline_rl.mode == 'train':
        files_path = cfg.nocturne_waymo_train_folder
        output_path = cfg.offline_rl.output_data_folder_train
    elif cfg.offline_rl.mode == 'val':
        files_path = cfg.nocturne_waymo_val_folder
        output_path = cfg.offline_rl.output_data_folder_val
    else:
        files_path = cfg.nocturne_waymo_val_interactive_folder 
        output_path = cfg.offline_rl.output_data_folder_val_interactive
    
    with open(os.path.join(files_path, 'valid_files.json')) as file:
        valid_veh_dict = json.load(file)
        files = list(valid_veh_dict.keys())
        # sort the files so that we have a consistent order
        files = sorted(files)

    chunk = list(range(cfg.offline_rl.chunk_idx * cfg.offline_rl.chunk_size, (cfg.offline_rl.chunk_idx + 1) * cfg.offline_rl.chunk_size))
    if len(files) < chunk[0]:
        raise ValueError("chunk_idx is too large for dataset size.")
    elif len(files) < chunk[-1]:
        chunk = [c for c in chunk if c < len(files)]

    k_disks_vocab = load_k_disks_vocab(cfg)
    collect_data(cfg=cfg, 
                dt=cfg.nocturne.dt, 
                steps=cfg.nocturne.steps, 
                output_path=output_path,
                files_path=files_path, 
                files=files,
                chunk=chunk,
                k_disks_vocab=k_disks_vocab)

if __name__ == '__main__':
    main()

import numpy as np
import maze_env
import maze
import cv2

def gen_data_range(env, samp_range, samp_size):
    color_list = []
    depth_list = []
    pose_list = []

    state, info_query = env.reset()
    center_x, center_y = env.agent_info["x"], env.agent_info["y"]

    count = 0
    while True:
        y = center_y + np.random.rand() * samp_range
        x = center_x + np.random.rand() * samp_range
        th = np.pi * 2 * np.random.rand()
        agent_info = {"x":x, "y":y, "theta":th}
        if env.maze.collision_detect(agent_info) == False:
            state, info = env.reset(gen_maze=False, init_agent_info=agent_info)
            color_list.append(np.expand_dims(info["color"],0))
            depth_list.append(np.expand_dims(info["depth"],0))
            pose_list.append(np.expand_dims(info["pose"],0))
            count += 1
            if count >= samp_size:
                color_list.append(np.expand_dims(info_query["color"],0))
                depth_list.append(np.expand_dims(info_query["depth"],0))
                pose_list.append(np.expand_dims(info_query["pose"],0))

                color_np = np.concatenate(color_list, 0)
                depth_np = np.concatenate(depth_list, 0)
                pose_np = np.concatenate(pose_list, 0)
                return color_np, depth_np, pose_np
            
def gen_dataset(env, scene_size, samp_range=2, samp_size=16):
    color_data, depth_data, pose_data = [], [], []
    for i in range(scene_size):
        print(i)
        color_np, depth_np, pose_np = gen_data_range(env, samp_range=samp_range, samp_size=samp_size)
        color_data.append(np.expand_dims(color_np, 0))
        depth_data.append(np.expand_dims(depth_np, 0))
        pose_data.append(np.expand_dims(pose_np, 0))
    
    color_data_np = np.concatenate(color_data, 0)
    depth_data_np = np.concatenate(depth_data, 0)
    pose_data_np = np.concatenate(pose_data, 0)

    return color_data_np, depth_data_np, pose_data_np

if __name__ == "__main__":
    # Select maze type.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', '-t', nargs='?', type=str, default="MazeBoardRandom", help='Maze type.')
    maze_type = parser.parse_args().type
    if maze_type == "MazeGridRoom":
        maze_obj = maze.MazeGridRoom()
    elif maze_type == "MazeGridRandom":
        maze_obj = maze.MazeGridRandom()
    elif maze_type == "MazeGridDungeon":
        maze_obj = maze.MazeGridDungeon()
    elif maze_type == "MazeBoardRoom":
        maze_obj = maze.MazeBoardRoom()
    elif maze_type == "MazeBoardRandom":
        maze_obj = maze.MazeBoardRandom()
    else:
        maze_obj = maze.MazeBoardRandom()

    # Initial Env
    env = maze_env.MazeBaseEnv(maze_obj, render_res=(96,96))
    dataset_path = "../maze_dataset/"
    import os
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    for i in range(100):
        color_data_np, depth_data_np, pose_data_np = gen_dataset(env, scene_size=128, samp_size=16)
        np.savez(os.path.join(dataset_path, "MazeGridRoom11x11_dataset96_"+str(i).zfill(3)+".npz"), color=color_data_np, depth=depth_data_np, pose=pose_data_np)
    
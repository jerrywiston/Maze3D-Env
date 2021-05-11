import cv2
import glm
import numpy as np
import pyrender
import scene_render
import maze

class MazeBaseEnv:
    def __init__(self, maze_obj, render_res=(192,192), velocity=0.1, ang_velocity=np.pi/18):
        self.maze = maze_obj
        self.render_res = render_res
        self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES | pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.SHADOWS_ALL
        self.rend = pyrender.OffscreenRenderer(self.render_res[0],self.render_res[1])
        self.map_scale = 16
        self.velocity = velocity
        self.ang_velocity = ang_velocity
        
    def _gen_scene(self):
        floor_list, wall_list, obj_list = self.maze.parse()
        self.scene = scene_render.gen_scene(floor_list, wall_list, obj_list)
        camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0, aspectRatio=1.0)
        self.camera_node = pyrender.Node(camera=camera)
        self.scene.add_node(self.camera_node)
    
    def _draw_traj(self, map_img):
        map_img = cv2.flip(map_img.copy(), 0)
        for i in range(len(self.traj)-1):
            x1 = int(self.map_scale*self.traj[i]["x"])
            y1 = int(self.map_scale*self.traj[i]["y"])
            x2 = int(self.map_scale*self.traj[i+1]["x"])
            y2 = int(self.map_scale*self.traj[i+1]["y"])
            cv2.line(map_img, (x1,y1), (x2,y2), (0,255,0), 1)
        map_img = cv2.flip(map_img, 0)
        return map_img
    
    def render_frame(self, toRGB=True):
        m = scene_render.get_cam_pose(self.agent_info['x'], self.agent_info['y'], self.agent_info['theta'])
        self.scene.set_pose(self.camera_node, m)
        color, depth = self.rend.render(self.scene, self.render_flags)
        if toRGB:
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        return color, depth
        
    def reset(self, gen_maze=True, init_agent_info=None):
        if gen_maze:
            self.maze.generate()
            self._gen_scene()
        # Set Camera 
        if init_agent_info is None:
            self.agent_info = self.maze.random_pose()
        else:
            self.agent_info = init_agent_info
        self.traj = [self.agent_info]
        color, depth = self.render_frame()
        # Return State / Info
        self.state = color
        map_img = self._draw_traj(self.maze.get_map(self.agent_info, self.map_scale))
        self.info = {"color":color, "depth":depth, "pose":self.agent_info, "map":map_img}
        return self.state, self.info
    
    def step(self, action):
        agent_info_new = self.agent_info.copy()
        if action == 0: # Forward (W)
            agent_info_new["x"] += self.velocity*np.cos(agent_info_new["theta"])
            agent_info_new["y"] += self.velocity*np.sin(agent_info_new["theta"])
        if action == 1: # Turn Left (Q)
            agent_info_new["theta"] += self.ang_velocity
        if action == 2: # Turn Right (E)
            agent_info_new["theta"] -= self.ang_velocity
        if action == 3: # Backward (S)
            agent_info_new["x"] -= self.velocity*np.cos(agent_info_new["theta"])
            agent_info_new["y"] -= self.velocity*np.sin(agent_info_new["theta"])
        if action == 4: # Shift Left (A)
            agent_info_new["x"] -= self.velocity*np.sin(agent_info_new["theta"])
            agent_info_new["y"] += self.velocity*np.cos(agent_info_new["theta"])
        if action == 5: # Shift Right (D)
            agent_info_new["x"] += self.velocity*np.sin(agent_info_new["theta"])
            agent_info_new["y"] -= self.velocity*np.cos(agent_info_new["theta"])
        
        if not self.maze.collision_detect(agent_info_new):
            self.agent_info = agent_info_new
        
        self.traj.append(self.agent_info)
        color, depth = self.render_frame()
        
        # Return State / Reward / Done / Info
        self.state_next = color
        map_img = self._draw_traj(self.maze.get_map(self.agent_info, self.map_scale))
        self.info = {"color":color, "depth":depth, "pose":self.agent_info, "map":map_img}
        return self.state_next, 0, False, self.info

    def render(self, res=(192,192)):
        color = cv2.resize(self.info["color"], res, interpolation=cv2.INTER_NEAREST) 
        depth = (cv2.resize(self.info["depth"]/5, res, res, interpolation=cv2.INTER_NEAREST) * 255)
        depth[depth>255] = 255
        depth = depth.astype(np.uint8)[...,np.newaxis]
        depth = np.concatenate([depth, depth, depth], 2)
        map_img = cv2.resize(self.info["map"], res, res, interpolation=cv2.INTER_NEAREST)
        render_img = cv2.hconcat([color, depth, map_img])
        cv2.imshow("MazeEnv", render_img)

class MazeNavEnv(MazeBaseEnv):
    def __init__(self, maze_obj, render_res=(192,192)):
        super().__init__(maze_obj, render_res)
        
    def _gen_scene(self):
        floor_list, wall_list, obj_list = self.maze.parse()
        obj_list.append({"voff":(self.nav_target["x"], self.nav_target["y"], 0.5), "color_id":0, "mesh_id":2, "scale":0.5})
        self.scene = scene_render.gen_scene(floor_list, wall_list, obj_list)
        camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0, aspectRatio=1.0)
        self.camera_node = pyrender.Node(camera=camera)
        self.scene.add_node(self.camera_node)
    
    def _draw_nav_target(self, map_img):
        map_img = cv2.flip(map_img.copy(), 0)
        x = int(self.map_scale*self.nav_target["x"])
        y = int(self.map_scale*self.nav_target["y"])
        cv2.circle(map_img, (x,y), int(0.2*self.map_scale), (255,255,0), 2)
        map_img = cv2.flip(map_img, 0)
        return map_img

    def reset(self, gen_maze=True, change_nav_target=True, init_agent_info=None):
        if gen_maze:
            self.maze.generate()
        
        if change_nav_target or gen_maze or not hasattr(self, "nav_target"):
            self.nav_target = self.maze.random_pose()

        self._gen_scene()

        # Set Camera 
        if init_agent_info is None:
            self.agent_info = self.maze.random_pose()
        else:
            self.agent_info = init_agent_info
        self.traj = [self.agent_info]
        color, depth = self.render_frame()
        # Return State / Info
        self.state = color
        map_img = self._draw_traj(self.maze.get_map(self.agent_info, self.map_scale))
        map_img = self._draw_nav_target(map_img)
        self.info = {"color":color, "depth":depth, "pose":self.agent_info, "map":map_img, "target":self.nav_target}
        return self.state, self.info

    def step(self, action):
        state_next, reward, done, info = super().step(action)
        self.info["map"] = self._draw_nav_target(self.info["map"])
        dist = (self.agent_info["x"] - self.nav_target["x"])**2 + (self.agent_info["y"] - self.nav_target["y"])**2
        dist = np.sqrt(dist)
        self.info["target"] = self.nav_target
        if dist < 0.5:
            done = True
            reward = 1.0
        return self.state_next, reward, done, self.info 
        
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
    #env = MazeBaseEnv(maze_obj, render_res=(192,192), velocity=0.4, ang_velocity=np.pi/18*4)
    env = MazeNavEnv(maze_obj, render_res=(192,192))
    state, info = env.reset()
    env.render()

    while(True):
        # Control Handle
        k = cv2.waitKey(0)
        run_step = False
        # Exit
        if k == 27:
            break
        # Reset Maze
        if k == 13:
            state, info = env.reset()
            env.render()
            continue
        # Reset Pose
        if k == 32:
            state, info = env.reset(gen_maze=False)
            env.render()
            continue
        if k == ord('w'):
            action = 0
            run_step = True
        if k == ord('s'):
            action = 3
            run_step = True
        if k == ord('q'):
            action = 1
            run_step = True
        if k == ord('e'):
            action = 2
            run_step = True
        if k == ord('a'):
            action = 4
            run_step = True
        if k == ord('d'):
            action = 5
            run_step = True
        
        if run_step:
            state_next, reward, done, info = env.step(action)
            print("\r", info["pose"], reward, done, end="\t")
            env.render()
            state = state_next.copy()
        
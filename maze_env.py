import cv2
import glm
import numpy as np
import pyrender
import scene_render
import maze

class BaseEnv:
    def __init__(self):
        pass

    def reset(self):
        return NotImplementedError
    
    def step(self, a):
        return NotImplementedError

class MazeBaseEnv(BaseEnv):
    def __init__(self, maze_obj, render_res=(192,192)):
        super().__init__()
        self.maze = maze_obj
        self._gen_scene()
        self.render_res = render_res
        self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES | pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.SHADOWS_ALL
        self.rend = pyrender.OffscreenRenderer(self.render_res[0],self.render_res[1])
        
    def _gen_scene(self):
        self.maze.generate()
        floor_list, wall_list, obj_list = self.maze.parse()
        self.scene = scene_render.gen_scene(floor_list, wall_list, obj_list)
        camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0, aspectRatio=1.0)
        self.camera_node = pyrender.Node(camera=camera)
        self.scene.add_node(self.camera_node)
    
    def _render_frame(self, toRGB=True):
        m = scene_render.get_cam_pose(self.agent_info['x'], self.agent_info['y'], self.agent_info['theta'])
        self.scene.set_pose(self.camera_node, m)
        color, depth = self.rend.render(self.scene, self.render_flags)
        if toRGB:
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        return color, depth

    def reset(self, gen_maze=True):
        if gen_maze:
            self._gen_scene()
        # Set Camera 
        self.agent_info = self.maze.random_pose()
        color, depth = self._render_frame()
        # Return State / Info
        return color, {"color":color, "depth":depth, "pose":self.agent_info, "map":self.maze.get_map(self.agent_info)}
    
    def step(self, action):
        agent_info_new = self.agent_info.copy()
        if action == 0: # Forward (W)
            agent_info_new["x"] += 0.1*np.cos(agent_info_new["theta"])
            agent_info_new["y"] += 0.1*np.sin(agent_info_new["theta"])
        if action == 1: # Backward (S)
            agent_info_new["x"] -= 0.1*np.cos(agent_info_new["theta"])
            agent_info_new["y"] -= 0.1*np.sin(agent_info_new["theta"])
        if action == 2: # Turn Left (Q)
            agent_info_new["theta"] += np.pi/18
        if action == 3: # Turn Rignt (E)
            agent_info_new["theta"] -= np.pi/18
        if action == 4: # Shift Left (A)
            agent_info_new["x"] -= 0.1*np.sin(agent_info_new["theta"])
            agent_info_new["y"] += 0.1*np.cos(agent_info_new["theta"])
        if action == 5: # Shift Right (D)
            agent_info_new["x"] += 0.1*np.sin(agent_info_new["theta"])
            agent_info_new["y"] -= 0.1*np.cos(agent_info_new["theta"])
        
        if not self.maze.collision_detect(agent_info_new):
            self.agent_info = agent_info_new
        
        color, depth = self._render_frame()
        # Return State / Reward / Done / Info
        return color, 0, False, {"color":color, "depth":depth, "pose":self.agent_info, "map":self.maze.get_map(self.agent_info)}
        
if __name__ == "__main__":
    #maze_obj = maze.MazeGridRoom()
    #maze_obj = maze.MazeGridRandom()
    #maze_obj = maze.MazeGridDungeon()
    #maze_obj = maze.MazeBoardRoom()
    maze_obj = maze.MazeBoardRandom()
    env = MazeBaseEnv(maze_obj, render_res=(192,192))
    state, info = env.reset()

    cv2.imshow("color", state)
    cv2.imshow("depth", info["depth"]/5)
    cv2.imshow("map", info["map"])

    while(True):
        # Control Handle
        k = cv2.waitKey(1)
        run_step = False
        if k == 27:
            break
        if k == ord('w'):
            action = 0
            run_step = True
        if k == ord('s'):
            action = 1
            run_step = True
        if k == ord('q'):
            action = 2
            run_step = True
        if k == ord('e'):
            action = 3
            run_step = True
        if k == ord('a'):
            action = 4
            run_step = True
        if k == ord('d'):
            action = 5
            run_step = True
        
        if run_step:
            state, reward, done, info = env.step(action)
            print("\r", info["pose"], end="\t")
            cv2.imshow("color", state)
            cv2.imshow("depth", info["depth"]/5)
            cv2.imshow("map", info["map"])
        
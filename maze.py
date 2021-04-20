import numpy as np
import cv2

class Maze:
    def __init__(self):
        self.maze = None
    
    def generate(self):
        raise NotImplementedError

    def parse(self):
        raise NotImplementedError
    
    def get_map(self):
        raise NotImplementedError
    
    def collision_detect(self):
        raise NotImplementedError

class MazeRoom(Maze):
    def __init__(self):
        super().__init__()
    
    def generate(self, room_size=(4,4)):
        self.maze = np.ones((room_size[0]+2, room_size[1]+2), dtype=np.uint8)
        self.maze[:,0] = 255
        self.maze[0,:] = 255
        self.maze[room_size[0]+1,:] = 255
        self.maze[:,room_size[1]+1] = 255

    def parse(self, obj_prob=0.2):
        floor_list = []
        wall_list = []
        obj_list = []
        for j in range(1,self.maze.shape[0]-1):
            for i in range(1,self.maze.shape[1]-1):
                if self.maze[j,i] < 255:
                    # Build Floor
                    floor_list.append({"voff":(i,j), "id":self.maze[j,i]})
                    # Build Wall Buttom
                    if self.maze[j-1,i]==255:
                        wall_list.append({"voff":(i,j), "id":self.maze[j,i], "type":"B"})
                    # Build Wall Top
                    if self.maze[j+1,i]==255:
                        wall_list.append({"voff":(i,j), "id":self.maze[j,i], "type":"T"})
                    # Build Wall Left
                    if self.maze[j,i-1]==255:
                        wall_list.append({"voff":(i,j), "id":self.maze[j,i], "type":"L"})
                    # Build Wall Right
                    if self.maze[j,i+1]==255:
                        wall_list.append({"voff":(i,j), "id":self.maze[j,i], "type":"R"})
                    # Build Object

                if 0 < self.maze[j,i] < 255 and np.random.rand() < obj_prob:
                    obj_list.append({"voff":(i+0.5,j+0.5,0.5)})

        return floor_list, wall_list, obj_list
    
    def get_map(self, agent_info, map_scale=16):
        x, y, th = agent_info["x"], agent_info["y"], agent_info["theta"]
        maze_re = cv2.cvtColor(self.maze, cv2.COLOR_GRAY2RGB)
        maze_re = 255-cv2.resize(maze_re, (self.maze.shape[1]*map_scale, self.maze.shape[0]*map_scale), interpolation=cv2.INTER_NEAREST)
        maze_draw = maze_re.copy()
        
        temp_y = int(y*map_scale)
        temp_x = int(x*map_scale)
        cv2.circle(maze_draw, (temp_x, temp_y), 4, (255,0,0), 3)
        temp_y2 = int((y + 0.4*np.sin(th)) * map_scale)
        temp_x2 = int((x + 0.4*np.cos(th)) * map_scale)
        cv2.line(maze_draw, (temp_x, temp_y), (temp_x2, temp_y2), (0,0,255), 3)
        maze_draw = cv2.flip(maze_draw,0)

        return maze_draw
    
    def collision_detect(self, agent_info, eps=0.1):
        collision = False
        if self.maze[int(agent_info['y']),int(agent_info['x'])] == 255:
            collision = True
        elif self.maze[int(agent_info['y']+eps),int(agent_info['x'])] == 255:
            collision = True
        elif self.maze[int(agent_info['y']-eps),int(agent_info['x'])] == 255:
            collision = True
        elif self.maze[int(agent_info['y']),int(agent_info['x']+eps)] == 255:
            collision = True
        elif self.maze[int(agent_info['y']),int(agent_info['x']-eps)] == 255:
            collision = True
        return collision

class MazeRandom(MazeRoom):
    def __init__(self):
        super().__init__()

    def generate(self, size=(11,11), room_max=(5,5), prob=0.8):
        import maze_gen
        self.maze = maze_gen.gen_maze(size[0],size[1],room_max,prob)

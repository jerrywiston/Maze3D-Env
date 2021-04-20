import numpy as np
import cv2

class Maze:
    def __init__(self):
        pass
    
    def generate(self):
        raise NotImplementedError

    def parse(self):
        raise NotImplementedError
    
    def get_map(self):
        raise NotImplementedError
    
    def collision_detect(self):
        raise NotImplementedError

#######################################
# Grid Maze
#######################################
class MazeGrid(Maze):
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
        cv2.circle(maze_draw, (temp_x, temp_y), int(map_scale/4), (255,0,0), 3)
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

class MazeGridRoom(MazeGrid):
    def generate(self, room_size=(4,4)):
        self.maze = np.ones((room_size[0]+2, room_size[1]+2), dtype=np.uint8)
        self.maze[:,0] = 255
        self.maze[0,:] = 255
        self.maze[room_size[0]+1,:] = 255
        self.maze[:,room_size[1]+1] = 255

class MazeGridRandom(MazeGrid):
    def generate(self, size=(11,11), room_max=(5,5), prob=0.8):
        import maze_gen
        self.maze = maze_gen.gen_maze(size[0],size[1],room_max,prob)

class MazeGridDungeon(MazeGrid):
    def generate(self):
        import dungeon
        gen = dungeon.Generator(width=24, height=24, max_rooms=5, min_room_xy=3, \
            max_room_xy=8, rooms_overlap=False, random_connections=1, random_spurs=3)
        gen.gen_level()
        self.maze = np.array(gen.level, dtype=np.uint8)

#######################################
# Board Maze
#######################################
class MazeBoard(Maze):
    def parse(self):
        floor_list = []
        wall_list = []
        obj_list = []
        for j in range(self.maze.shape[0]):
            for i in range(self.maze.shape[1]):
                floor_list.append({"voff":(i,j), "id":0})
                # Build Wall Buttom
                if self.maze[j,i,2]==1:
                    wall_list.append({"voff":(i,j), "id":0, "type":"B"})
                # Build Wall Top
                if self.maze[j,i,0]==1:
                    wall_list.append({"voff":(i,j), "id":0, "type":"T"})
                # Build Wall Left
                if self.maze[j,i,1]==1:
                    wall_list.append({"voff":(i,j), "id":0, "type":"L"})
                # Build Wall Right
                if self.maze[j,i,3]==1:
                    wall_list.append({"voff":(i,j), "id":0, "type":"R"})

        return floor_list, wall_list, obj_list
    
    def get_map(self, agent_info, map_scale=16, wall_rate=0.4):
        x, y, th = agent_info["x"], agent_info["y"], agent_info["theta"]
        maze_draw  = 255*np.ones((self.maze.shape[0]*map_scale, self.maze.shape[1]*map_scale, 3), dtype=np.uint8)
        
        # Draw Wall
        wall_size = int(map_scale*wall_rate)
        for j in range(self.maze.shape[0]):
            for i in range(self.maze.shape[1]):
                if self.maze[j,i,0] == 1:
                    x1 = (i)*map_scale
                    y1 = (j+1)*map_scale
                    x2 = (i+1)*map_scale
                    y2 = (j+1)*map_scale
                    cv2.line(maze_draw, (x1, y1), (x2, y2), (0,0,0), wall_size)
                if self.maze[j,i,1] == 1:
                    x1 = (i)*map_scale
                    y1 = (j)*map_scale
                    x2 = (i)*map_scale
                    y2 = (j+1)*map_scale
                    cv2.line(maze_draw, (x1, y1), (x2, y2), (0,0,0), wall_size)
                if self.maze[j,i,2] == 1:
                    x1 = (i)*map_scale
                    y1 = (j)*map_scale
                    x2 = (i+1)*map_scale
                    y2 = (j)*map_scale
                    cv2.line(maze_draw, (x1, y1), (x2, y2), (0,0,0), wall_size)
                if self.maze[j,i,3] == 1:
                    x1 = (i+1)*map_scale
                    y1 = (j)*map_scale
                    x2 = (i+1)*map_scale
                    y2 = (j+1)*map_scale
                    cv2.line(maze_draw, (x1, y1), (x2, y2), (0,0,0), wall_size)

        # Draw Agent
        temp_y = int(y*map_scale)
        temp_x = int(x*map_scale)
        cv2.circle(maze_draw, (temp_x, temp_y), int(map_scale/4), (255,0,0), 3)
        temp_y2 = int((y + 0.4*np.sin(th)) * map_scale)
        temp_x2 = int((x + 0.4*np.cos(th)) * map_scale)
        cv2.line(maze_draw, (temp_x, temp_y), (temp_x2, temp_y2), (0,0,255), 3)
        maze_draw = cv2.flip(maze_draw,0)
        return maze_draw
    
    def collision_detect(self, agent_info):
        return False

# T, L, B, R
class MazeBoardRoom(MazeBoard):
    def generate(self, room_size=(4,4)):
        self.maze = np.zeros((room_size[0], room_size[1], 4), dtype=np.uint8)
        self.maze[0,:,2] = 1
        self.maze[room_size[0]-1,:,0] = 1
        self.maze[:,0,1] = 1
        self.maze[:,room_size[1]-1,3] = 1
        
if __name__ == "__main__":
    maze_obj = MazeBoardRoom()
    maze_obj.generate()
    print(maze_obj.maze)
    maze_draw = maze_obj.get_map({"x":2, "y":2, "theta":0})
    cv2.imshow("test", maze_draw)
    cv2.waitKey(0)

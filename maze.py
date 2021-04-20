class Maze:
    def __init__():
        pass

    def parse_maze():
        raise NotImplementedError
    
    def get_map():
        raise NotImplementedError

def parse_maze(maze):
    floor_list = []
    wall_list = []
    for j in range(1,maze.shape[0]-1):
        for i in range(1,maze.shape[1]-1):
            if maze[j,i] < 255:
                # Build Floor
                floor_list.append({"voff":(i,j), "id":maze[j,i]})
                # Build Wall Buttom
                if maze[j-1,i]==255:
                    wall_list.append({"voff":(i,j), "id":maze[j,i], "type":"B"})
                # Build Wall Top
                if maze[j+1,i]==255:
                    wall_list.append({"voff":(i,j), "id":maze[j,i], "type":"T"})
                # Build Wall Left
                if maze[j,i-1]==255:
                    wall_list.append({"voff":(i,j), "id":maze[j,i], "type":"L"})
                # Build Wall Right
                if maze[j,i+1]==255:
                    wall_list.append({"voff":(i,j), "id":maze[j,i], "type":"R"})
    return floor_list, wall_list

def gen_obj_mesh(maze, gen_prob=0.2):
    path_obj = './resource/texture/obj'
    flist_obj = os.listdir(path_obj)
    path_mesh = './resource/mesh'
    flist_mesh = os.listdir(path_mesh)
    struct_obj = {'v':[], 'vn':[], 'f':[], 'vt':[], 'foff':0}
    num_obj = 0
    for j in range(1,maze.shape[0]-1):
        for i in range(1,maze.shape[1]-1):
            # Check in the room
            if 0 < maze[j,i] < 255 and np.random.rand() < gen_prob:
                mesh_id = np.random.randint(len(flist_mesh))
                color_id = np.random.randint(len(flist_obj))
                scale = np.random.uniform(0.2,0.6)
                v, vn, f, vt, foff = obj_loader.load_(os.path.join(path_mesh, flist_mesh[mesh_id]), \
                            (i+0.5,j+0.5,0.5), struct_obj['foff'], scale, len(flist_obj), color_id)
                struct_obj = add_struct(v, vn, f, vt, foff, struct_obj)
                num_obj += 1
    #print(num_obj)
    if num_obj == 0:
        return None
    
    image_obj = read_texture(path_obj, flist_obj, 256)
    material_obj = trimesh.visual.texture.SimpleMaterial(image=image_obj)
    color_visuals_obj = trimesh.visual.TextureVisuals(uv=struct_obj['vt'], image=image_obj, material=material_obj)

    mesh_obj = trimesh.Trimesh(vertices=struct_obj['v'], faces=struct_obj['f'], visual=color_visuals_obj)
    mesh_obj_pr = pyrender.Mesh.from_trimesh(mesh_obj)
    return mesh_obj_pr

def gen_obj_node(maze, gen_prob=0.2):
    path_obj = './resource/texture/obj'
    flist_obj = os.listdir(path_obj)
    path_mesh = './resource/mesh'
    flist_mesh = os.listdir(path_mesh)
    num_obj = 0
    obj_list = []
    for j in range(1,maze.shape[0]-1):
        for i in range(1,maze.shape[1]-1):
            # Check in the room
            if 0 < maze[j,i] < 255 and np.random.rand() < gen_prob:
                struct_obj = {'v':[], 'vn':[], 'f':[], 'vt':[], 'foff':0}
                mesh_id = np.random.randint(len(flist_mesh))
                color_id = np.random.randint(len(flist_obj))
                scale = np.random.uniform(0.2,0.6)
                v, vn, f, vt, foff = obj_loader.load_(os.path.join(path_mesh, flist_mesh[mesh_id]), \
                            (0,0,0), struct_obj['foff'], scale, len(flist_obj), color_id)
                struct_obj = add_struct(v, vn, f, vt, foff, struct_obj)
                
                rot = np.random.uniform(0,np.pi*2)
                r = glm.mat4_cast(glm.quat(glm.vec3(0,0,rot)))
                t = glm.translate(glm.mat4(1), glm.vec3(i+0.5,j+0.5,scale*0.5))
                m = r * glm.transpose(t)
                m = np.array(m)
                
                image_obj = read_texture(path_obj, flist_obj, 256)
                material_obj = trimesh.visual.texture.SimpleMaterial(image=image_obj)
                color_visuals_obj = trimesh.visual.TextureVisuals(uv=struct_obj['vt'], image=image_obj, material=material_obj)
                mesh_obj = trimesh.Trimesh(vertices=struct_obj['v'], faces=struct_obj['f'], visual=color_visuals_obj)
                mesh_obj_pr = pyrender.Mesh.from_trimesh(mesh_obj)

                obj = pyrender.Node(mesh=mesh_obj_pr, matrix=m)
                obj_list.append(obj)
                num_obj += 1
    #print(num_obj)
    if num_obj == 0:
        return None
    obj_node = pyrender.Node(children=obj_list)
    return obj_node
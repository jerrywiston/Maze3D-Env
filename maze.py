##############################################
# Multiple textures, solve the sampling problem.
##############################################
import os
import trimesh
import pyrender
import numpy as np
import glm
import cv2
import maze_gen
import structure

def add_struct(v, vn, f, uv, foff, struct_obj):
    struct_obj['v'] += v
    struct_obj['vn'] += vn
    struct_obj['f'] += f
    struct_obj['uv'] += uv
    struct_obj['foff'] += foff
    return struct_obj

def read_texture(path, flist, size=256):
    img_tex = None
    for fname in flist:
        img = cv2.imread(os.path.join(path,fname))
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        if img_tex is None:
            # Duplicate to solbe the sampling problem.
            img_tex = cv2.hconcat([img, img, img])
        else:
            # Combine multiple textures.
            img_tex = cv2.hconcat([img_tex, img, img, img])

    img_tex = cv2.cvtColor(img_tex, cv2.COLOR_RGB2BGR)
    return img_tex

def gen_maze_mesh(maze):
    path_floor = './resource/texture/floor'
    flist_floor = os.listdir(path_floor)
    path_wall = './resource/texture/wall'
    flist_wall = os.listdir(path_wall)

    struct_floor = {'v':[], 'vn':[], 'f':[], 'uv':[], 'foff':0}
    struct_wall = {'v':[], 'vn':[], 'f':[], 'uv':[], 'foff':0}
    for j in range(1,maze.shape[0]-1):
        for i in range(1,maze.shape[1]-1):
            if maze[j,i] < 255:
                # Build Floor
                v, vn, f, uv, foff = structure.get_struct_floor((j,i), struct_floor['foff'], 0, len(flist_floor))
                struct_floor = add_struct(v, vn, f, uv, foff, struct_floor)
                # Build Wall Buttom
                if maze[j+1,i]==255:
                    v, vn, f, uv, foff = structure.get_struct_wall_buttom((j,i), struct_wall['foff'], maze[j,i], len(flist_wall))
                    struct_wall = add_struct(v, vn, f, uv, foff, struct_wall)
                # Build Wall Top
                if maze[j-1,i]==255:
                    v, vn, f, uv, foff = structure.get_struct_wall_top((j,i), struct_wall['foff'], maze[j,i], len(flist_wall))
                    struct_wall = add_struct(v, vn, f, uv, foff, struct_wall)
                # Build Wall Left
                if maze[j,i-1]==255:
                    v, vn, f, uv, foff = structure.get_struct_wall_left((j,i), struct_wall['foff'], maze[j,i], len(flist_wall))
                    struct_wall = add_struct(v, vn, f, uv, foff, struct_wall)
                # Build Wall Right
                if maze[j,i+1]==255:
                    v, vn, f, uv, foff = structure.get_struct_wall_right((j,i), struct_wall['foff'], maze[j,i], len(flist_wall))
                    struct_wall = add_struct(v, vn, f, uv, foff, struct_wall)

    tex_size = 256
    # Texture Floor
    image_floor = read_texture(path_floor, flist_floor, tex_size)
    material_floor = trimesh.visual.texture.SimpleMaterial(image=image_floor)
    color_visuals_floor = trimesh.visual.TextureVisuals(uv=struct_floor['uv'], image=image_floor, material=material_floor)
    # Mesh Floor
    mesh_floor = trimesh.Trimesh(vertices=struct_floor['v'], faces=struct_floor['f'], visual=color_visuals_floor)
    mesh_floor_pr = pyrender.Mesh.from_trimesh(mesh_floor)

    # Texture Wall
    image_wall = read_texture(path_wall, flist_wall, tex_size)
    material_wall = trimesh.visual.texture.SimpleMaterial(image=image_wall)
    color_visuals_wall = trimesh.visual.TextureVisuals(uv=struct_wall['uv'], image=image_wall, material=material_wall)
    # Mesh Wall
    mesh_wall = trimesh.Trimesh(vertices=struct_wall['v'], faces=struct_wall['f'], visual=color_visuals_wall)
    mesh_wall_pr = pyrender.Mesh.from_trimesh(mesh_wall)

    return mesh_floor_pr, mesh_wall_pr

def gen_obj_mesh(maze, gen_prob=0.2):
    for j in range(1,maze.shape[0]-1):
        for i in range(1,maze.shape[1]-1):
            # Check in the room
            if 0 < maze[j,i] < 255 and np.random.rand() < gen_prob:
                mesh_obj = trimesh.load('./resource/obj/Cylinder.obj')
                mesh_obj_pr = pyrender.Mesh.from_trimesh(mesh_floor)

def gen_scene(w=11, h=11, room_max=(5,5), prob=0.5):
    # Generate Maze
    maze = maze_gen.gen_maze(11,11,room_max,prob)
    # Pyrender
    mesh_floor_pr, mesh_wall_pr = gen_maze_mesh(maze)
    amb_intensity = 0.5
    bg_color = np.array([180,200,255,0])
    scene = pyrender.Scene(ambient_light=amb_intensity*np.ones(3), bg_color=bg_color)
    scene.add(mesh_floor_pr)
    scene.add(mesh_wall_pr)
    return scene, maze

def create_raymond_lights(intensity=1.0):
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=intensity),
            matrix=matrix
        ))

    return nodes

if __name__ == "__main__":
    # Camera
    scene, maze = gen_scene()
    agent_info = {"x":5, "y":5, "theta":0}
    camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0, aspectRatio=1.0)
    r = glm.mat4_cast(glm.quat(glm.vec3(-np.pi/2,0,0)))
    t = glm.translate(glm.mat4(1), glm.vec3(agent_info['x']+0.5,agent_info['y']+0.5,0.5))
    m = r * glm.transpose(t)
    camera_node = pyrender.Node(camera=camera, matrix=m)
    scene.add_node(camera_node)

    ############### Light ###############
    #light = pyrender.SpotLight(color=np.ones(3), intensity=6.0, innerConeAngle=np.pi/4.0, outerConeAngle=np.pi/2.0)
    #scene.add(light)
    #light_nodes = create_raymond_lights(intensity=5.0)
    #for light in light_nodes:
    #    scene.add_node(light)

    ############### Viewer ###############
    USE_VIEWER = False
    render_flags = { \
        "flip_wireframe":False, #default:False
        "all_wireframe":False,   #default:False
        "all_solid":False,      #default:False
        "shadows":True,         #default:False
        "face_notmals":False,   #default:False
        "cull_faces":False,     #default:True
        "point_size":1,         #default:1
    }
    if USE_VIEWER:
        pyrender.Viewer(scene, render_flags=render_flags, use_raymond_lighting=True)
    
    ############### Off-Screen Render ###############
    import time
    render_frame = True
    while(True):
        print("\r", agent_info['x'], agent_info['y'], agent_info['theta']*180/np.pi, end="")
        maze_re = cv2.cvtColor(maze, cv2.COLOR_GRAY2RGB)
        map_scale = 16
        maze_re = 255-cv2.resize(maze_re, (maze.shape[1]*map_scale, maze.shape[0]*map_scale), interpolation=cv2.INTER_NEAREST)
        maze_draw = maze_re.copy()
        temp_y = int((agent_info['y']+0.5)*map_scale)
        temp_x = int((agent_info['x']+0.5)*map_scale)
        cv2.circle(maze_draw, (temp_y, temp_x), 4, (255,0,0), 3)
        temp_y2 = int((agent_info["y"] + 0.5 + 0.4*np.cos(agent_info["theta"])) * map_scale)
        temp_x2 = int((agent_info["x"] + 0.5 + 0.4*np.sin(agent_info["theta"])) * map_scale)
        cv2.line(maze_draw, (temp_y, temp_x), (temp_y2, temp_x2), (0,0,255), 3)
        cv2.imshow("maze", maze_draw)

        r = glm.mat4_cast(glm.quat(glm.vec3(-np.pi/2,agent_info['theta'],0)))
        t = glm.translate(glm.mat4(1), glm.vec3(agent_info['x']+0.5,agent_info['y']+0.5,0.5))
        m = r * glm.transpose(t)
        m = np.array(m)
        scene.set_pose(camera_node, m)

        if render_frame:
            start = time.time()
            r = pyrender.OffscreenRenderer(256,256)
            flags = pyrender.RenderFlags.SKIP_CULL_FACES #| pyrender.RenderFlags.SHADOWS_ALL
            color, depth = r.render(scene, flags)
            end = time.time()
            print(" Time ", end - start)
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            cv2.imshow("camera", color)
            render_frame = False
    
        k = cv2.waitKey(1)
        if k == 27:
            break
        if k == ord('w'):
            agent_info["x"] += 0.1*np.sin(agent_info["theta"])
            agent_info["y"] += 0.1*np.cos(agent_info["theta"])
            render_frame = True
        if k == ord('s'):
            agent_info["x"] -= 0.1*np.sin(agent_info["theta"])
            agent_info["y"] -= 0.1*np.cos(agent_info["theta"])
            render_frame = True
        if k == ord('d'):
            agent_info["x"] += 0.1*np.cos(agent_info["theta"])
            agent_info["y"] -= 0.1*np.sin(agent_info["theta"])
            render_frame = True
        if k == ord('a'):
            agent_info["x"] -= 0.1*np.cos(agent_info["theta"])
            agent_info["y"] += 0.1*np.sin(agent_info["theta"])
            render_frame = True
        if k == ord('e'):
            agent_info["theta"] += np.pi/18
            render_frame = True
        if k == ord('q'):
            agent_info["theta"] -= np.pi/18
            render_frame = True
    print()
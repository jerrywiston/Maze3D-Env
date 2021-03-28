import trimesh
import pyrender
import numpy as np
import glm

fuze_trimesh = trimesh.load('./resource/obj/Cube.obj')
mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
scene = pyrender.Scene()

nm = pyrender.Node(mesh=mesh, matrix=np.eye(4))

q = glm.quat(glm.vec3(np.pi,np.pi/2,np.pi))
m = glm.mat4_cast(q)
pose2 = np.array(m)
print(pose2)
nm2 = pyrender.Node(mesh=mesh, matrix=pose2)
#scene.add(mesh)
scene.add_node(nm)
scene.add_node(nm2)
pyrender.Viewer(scene, use_raymond_lighting=True)
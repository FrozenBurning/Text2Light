# blender --background --python 04_principled_bsdf.py --render-frame 1 -- </path/to/output/image> <resolution_percentage> <num_samples>

import bpy
import sys
import math
import os

working_dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(working_dir_path)

import blender_utils


def set_principled_node_as_rough_white(principled_node: bpy.types.Node) -> None:
    blender_utils.set_principled_node(
        principled_node=principled_node,
        base_color=(0.95, 0.95, 0.95, 1.0),
        metallic=0.5,
        specular=0.5,
        roughness=0.9,
    )

def set_principled_node_as_shiny_blue(principled_node: bpy.types.Node) -> None:
    blender_utils.set_principled_node(
        principled_node=principled_node,
        base_color=(0.1, 0.2, 0.6, 1.0),
        metallic=1.0,
        specular=0.5,
        roughness=0.1,
    )

def set_principled_node_as_ceramic(principled_node: bpy.types.Node) -> None:
    blender_utils.set_principled_node(
        principled_node=principled_node,
        base_color=(0.8, 0.8, 0.8, 1.0),
        subsurface=0.1,
        subsurface_color=(0.9, 0.9, 0.9, 1.0),
        subsurface_radius=(1.0, 1.0, 1.0),
        metallic=0.2,
        specular=0.5,
        roughness=0.0,
    )


def set_principled_node_as_gold(principled_node: bpy.types.Node) -> None:
    blender_utils.set_principled_node(
        principled_node=principled_node,
        base_color=(1.00, 0.71, 0.22, 1.0),
        metallic=0.9,
        specular=0.5,
        roughness=0.273,
    )

def set_principled_node_as_silver(principled_node: bpy.types.Node) -> None:
    blender_utils.set_principled_node(
        principled_node=principled_node,
        base_color=(0.95, 0.95, 0.95, 1.0),
        metallic=1.0,
        specular=0.5,
        roughness=0.0,
    )


def set_principled_node_as_glass(principled_node: bpy.types.Node) -> None:
    blender_utils.set_principled_node(principled_node=principled_node,
                              base_color=(0.95, 0.95, 0.95, 1.0),
                              metallic=0.0,
                              specular=0.5,
                              roughness=0.0,
                              clearcoat=0.5,
                              clearcoat_roughness=0.030,
                              ior=1.45,
                              transmission=0.98)


def set_scene_objects() -> bpy.types.Object:
    left_object, center1_object, center2_object, right_object = blender_utils.create_three_smooth_sphere(radius=3)
    monkey = blender_utils.create_smooth_monkey(location=(0.0, 0.0, 11), rotation=(0.0, 0.0, +math.pi * 30.0 / 180.0))
    monkey.scale = (4,4,4)

    mat = blender_utils.add_material("Material_mon", use_nodes=True, make_node_tree_empty=True)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    set_principled_node_as_silver(principled_node)
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    monkey.data.materials.append(mat)

    mat = blender_utils.add_material("Material_Left", use_nodes=True, make_node_tree_empty=True)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    set_principled_node_as_glass(principled_node)
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    left_object.data.materials.append(mat)

    mat = blender_utils.add_material("Material_Center1", use_nodes=True, make_node_tree_empty=True)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    set_principled_node_as_rough_white(principled_node)
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    center1_object.data.materials.append(mat)

    mat = blender_utils.add_material("Material_Center2", use_nodes=True, make_node_tree_empty=True)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    set_principled_node_as_silver(principled_node)
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    center2_object.data.materials.append(mat)


    mat = blender_utils.add_material("Material_Right", use_nodes=True, make_node_tree_empty=True)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    set_principled_node_as_gold(principled_node)
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    right_object.data.materials.append(mat)

    current_object = blender_utils.create_plane(size=100.0, name="Floor")
    mat = blender_utils.add_material("Material_Plane", use_nodes=True, make_node_tree_empty=True)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    set_principled_node_as_ceramic(principled_node)
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    current_object.data.materials.append(mat)

    bpy.ops.object.empty_add(location=(0.0, -0.75, 6.3))
    focus_target = bpy.context.object
    return focus_target


# Args
input_hdri_path = str(sys.argv[sys.argv.index('--') + 4])
output_file_path = bpy.path.relpath(str(sys.argv[sys.argv.index('--') + 1]))
resolution_percentage = int(sys.argv[sys.argv.index('--') + 2])
num_samples = int(sys.argv[sys.argv.index('--') + 3])

# Parameters
# hdri_path = os.path.join(working_dir_path, "assets/HDRIs/green_point_park_2k.hdr")
if os.path.isdir(input_hdri_path):
    hdris = sorted(os.listdir(input_hdri_path))
else:
    hdris = [input_hdri_path]

valid_path = []
for hdri in hdris:
    if '.exr' in hdri or '.hdr' in hdri:
        path = os.path.join(input_hdri_path, hdri)
    else:
        continue
    valid_path.append(path)

# Scene Building
scene = bpy.data.scenes["Scene"]
world = scene.world

## Reset
blender_utils.clean_objects()

## Suzannes
focus_target = set_scene_objects()

## Camera
bpy.ops.object.camera_add(location=(0.0, -25.0, 2.0))
camera_object = bpy.context.object

blender_utils.add_track_to_constraint(camera_object, focus_target)
blender_utils.set_camera_params(camera_object.data, focus_target, lens=25, fstop=1.4)

blender_utils.set_cycles_renderer(scene, camera_object, num_samples)

blender_utils.build_environment_texture_background(world)

for hdri_path in valid_path:
    ## Lights
    blender_utils.set_environment_texture_background(world, hdri_path)

    name = hdri_path.split('/')[-1][:-4]
    # Render Setting
    blender_utils.set_output_properties(scene, resolution_percentage, os.path.join(output_file_path, name+'_balls.png'))
    bpy.ops.render.render(write_still=True)

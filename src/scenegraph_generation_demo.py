import numpy as np
import pandas as pd
import open3d as o3d
import yaml
import os

from LostFound.src import SceneGraph, get_scene_graph
from LostFound.src import preprocess_scan
from LostFound.src.utils import parse_txt
# The drawer_integration is only in the Spotlight repo!!
# from LostFound.src.data_processing.drawer_integration import parse_txt 


with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

if __name__ == "__main__":
    SCAN_DIR = f"{os.environ['LSARP']}/{config['path_to_data']}prescans/{config['pre_scanned_graphs']['high_res']}/"
    SAVE_DIR = f"{os.environ['LSARP']}/{config['path_to_data']}3D-Scene-Understanding"

    scene_graph = get_scene_graph(SCAN_DIR, categories_to_remove=["curtain", "door"], transform_to_spot_frame=True, drawers=False, light_switches=True)
    ### visualizes the current state of the scene graph with different visualizaion options:
    scene_graph.visualize(centroids=True, connections=True, labels=True)

    ### Add lamps to the scene:
    lamp_ids = []
    for idx,node in enumerate(scene_graph.nodes.values()):
        if node.sem_label == 28:
            lamp_ids.append(node.object_id)

    switch_ids = []
    for idx,node in enumerate(scene_graph.nodes.values()):
        if node.sem_label == 232:
            switch_ids.append(node.object_id)

    switch_idx_upper = switch_ids[0]
    switch_idx_lower = switch_ids[1]

    lamp_idx_upper = lamp_ids[0]
    lamp_idx_lower = lamp_ids[1]

    # e.g. add a lamp to a light switch: 
    scene_graph.nodes[switch_idx_upper].add_lamp(lamp_idx_upper)
    scene_graph.nodes[switch_idx_lower].add_lamp(lamp_idx_lower)
    
    # scene_graph.nodes[switch_idx_upper].button_count = 1
    # scene_graph.nodes[switch_idx_upper].interaction = "PUSH"
    # scene_graph.nodes[switch_idx_lower].button_count = 1
    # scene_graph.nodes[switch_idx_lower].interaction = "PUSH"
    
    # scene_graph.nodes[lamp_idx_upper].state = "ON"
    # scene_graph.nodes[lamp_idx_lower].state = "ON"

    scene_graph.visualize(centroids=True, labels=True, connections=True)

    # to save the point cloud as a .ply file:
    # scene_graph.save_ply(SCAN_DIR + "scene.ply")

    ### save the point clouds of the nodes to a .ply file:
    for node in scene_graph.nodes.values():
        pts = node.points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(f"{SAVE_DIR}/{scene_graph.label_mapping[node.sem_label]}_{node.object_id}.ply", pcd)
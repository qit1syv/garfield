from PIL import Image
import numpy as np
import viser
import viser.transforms as vtf
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from cuml.cluster.hdbscan import HDBSCAN
from PIL import Image
from garfield.llm_sud import generate_response
import numpy as np
import pickle
from pathlib import Path
from viser import transforms as tf

class TreeNode:
    def __init__(self, scale, mask):
        self.scale = scale
        self.id = np.random.randint(0, 1000000)
        self.mask = mask 
        self.children = []
        
class SemanticUnitDiscriminator:
    def __init__(self):
        pass

    def get_separate_object(self,  *args, **kwargs):
        """
        This method should be overridden by subclasses to implement specific
        functionality for separating objects based on the given model and mask.

        Returns:
        An object or data structure that represents the result of the separation operation.
        """
        raise NotImplementedError("Subclass must implement abstract method")

class SUD_Human(SemanticUnitDiscriminator):
    def __init__(self):
        super().__init__()
        self.type = "human"
    def get_separate_object(self):
        answer = input("Is the cluster a separate object? (y/n)")
        if answer == "y":
            return True
        return False

class SUD_VLM(SemanticUnitDiscriminator):
    def __init__(self):
        super().__init__()
        self.type = "vlm"
    def get_separate_object(self, raw_image, masked_image):
        raw_image_pil = Image.fromarray(np.uint8(raw_image * 255))
        masked_image_pil = Image.fromarray(np.uint8(masked_image * 255))
        target_scene_path = './prompt_examples/target_scene.png'
        target_scene_mask_path = './prompt_examples/target_scene_mask.png'
        raw_image_pil.save(target_scene_path)
        masked_image_pil.save(target_scene_mask_path)
        response_string = generate_response(target_scene_path, target_scene_mask_path)[:6].lower()
        if "yes" in response_string:
            return True
        return False


############################### Util Functions for the tree generation ############################################
def generate_candidate_clusters(grouping_model, positions, curr_scale):
    group_feats = grouping_model.get_grouping_at_points(positions, curr_scale).cpu().numpy()  # (N, 256)
    positions_numpy = positions.cpu().numpy()

    vec_o3d = o3d.utility.Vector3dVector(positions_numpy)
    pc_o3d = o3d.geometry.PointCloud(vec_o3d)
    min_bound = np.clip(pc_o3d.get_min_bound(), -1, 1)
    max_bound = np.clip(pc_o3d.get_max_bound(), -1, 1)
    downsample_size = 0.01 * curr_scale
    pc, _, ids = pc_o3d.voxel_down_sample_and_trace(
        max(downsample_size, 0.0001), min_bound, max_bound
    )
    id_vec = np.array([points[0] for points in ids])  # indices of gaussians kept after downsampling
    group_feats_downsampled = group_feats[id_vec]
    positions_downsampled = np.array(pc.points)
    clusterer = HDBSCAN(
        cluster_selection_epsilon=0.1,
        min_samples=30,
        min_cluster_size=30,
        allow_single_cluster=True,
    ).fit(group_feats_downsampled)

    non_clustered = np.ones(positions.shape[0], dtype=bool)
    non_clustered[id_vec] = False
    labels = clusterer.labels_.copy()
    clusterer.labels_ = -np.ones(positions.shape[0], dtype=np.int32)
    clusterer.labels_[id_vec] = labels

    # Assign the full gaussians to the spatially closest downsampled gaussian, with scipy NearestNeighbors.
    positions_np = positions[non_clustered]
    if positions_np.shape[0] > 0:  # i.e., if there were points removed during downsampling, which commonpositions happens
        k = 1
        nn_model = NearestNeighbors(
            n_neighbors=k, algorithm="auto", metric="euclidean"
        ).fit(positions_downsampled)
        _, indices = nn_model.kneighbors(positions_np.cpu()) # might be faster to do this on the gpu
        clusterer.labels_[non_clustered] = labels[indices[:, 0]]

    labels = clusterer.labels_
    noise_mask = labels == -1
    if noise_mask.sum() != 0 and (labels>=0).sum() > 0:
        # if there is noise, but not all of it is noise, relabel the noise
        valid_mask = labels >=0
        valid_positions = positions[valid_mask]
        k = 1
        nn_model = NearestNeighbors(
            n_neighbors=k, algorithm="auto", metric="euclidean"
        ).fit(valid_positions.cpu()) # might be faster to do this on the gpu
        noise_positions = positions[noise_mask]
        _, indices = nn_model.kneighbors(noise_positions.cpu())
        # for now just pick the closest cluster 
        noise_relabels = labels[valid_mask][indices[:, 0]]
        labels[noise_mask] = noise_relabels
        clusterer.labels_ = labels
    labels = clusterer.labels_
    return labels
    
def add_child(node, child):
    node.children.append(child)

def visualize_tree(scales, wxyzs, means, colors, opacities, root: TreeNode):        
    # visualize the tree in viser
    Rs = np.array([vtf.SO3(wxyz).as_matrix() for wxyz in wxyzs])
    covariances = np.einsum("nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs)
    server = viser.ViserServer(share=True)
    init_start_offset = np.array([0, 0, 2])
    node_offset = np.array([0, 0, -0.2])
    vertical_component_arrow = 1
    
    running_centers, running_covariances, running_rgbs, running_opacities = None, None, None, None
    
    def rescale_points_to_fit_in_cube(points, covariances, a):
        min_vals = points.min(axis=0)
        max_vals = points.max(axis=0)
        
        dims = max_vals - min_vals
        scale_factor = a / (np.max(dims))
        
        scaled_points = (points - min_vals) * scale_factor
        translation = -a / 2
        centered_at_origin_points = scaled_points + translation
        covariances = covariances * scale_factor **2
        return centered_at_origin_points, covariances
    
    def calculate_radius_and_points_on_cone_base(n, h):
        r = max(0.8, ((n * np.sqrt(2)) / (2 * np.pi)) + 0.2*n) # 0.2*n is to make the node unit cubes further apart
        points = []
        for i in range(n):
            theta = 2 * np.pi * i / n
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = -h
            points.append((x, y, z))
        return points
    
    def visualize_tree_helper(node: TreeNode, start_offset: np.ndarray = np.array([0, 0, 0]), depth: int = 0, parent_children_index: int = 0):
        nonlocal running_centers, running_covariances, running_rgbs, running_opacities
        if node is None:
            return
        indices = np.where(node.mask)
        rescaled_points, rescaled_covariances = rescale_points_to_fit_in_cube(means[indices], covariances[indices], 1)
        position = start_offset - rescaled_points.mean(axis=0)
        running_centers = rescaled_points + position if running_centers is None else np.concatenate((running_centers, rescaled_points + position))
        running_covariances = rescaled_covariances if running_covariances is None else np.concatenate((running_covariances, rescaled_covariances))
        running_rgbs = colors[indices] if running_rgbs is None else np.concatenate((running_rgbs, colors[indices]))
        running_opacities = opacities[indices]*2 if running_opacities is None else np.concatenate((running_opacities, opacities[indices]*2)) # multiply by 2 to make it more visible
        
        positions = np.zeros((2, 3))
        positions[0] = start_offset + node_offset
        if len(node.children) == 0:
            return
        points = calculate_radius_and_points_on_cone_base(len(node.children), vertical_component_arrow)
        for i, child in enumerate(node.children):
            positions[1] = positions[0] + np.array(points[i]) + np.array([0, 0, -i-depth*parent_children_index])
            server.add_spline_catmull_rom(f"/{node.id}_to_{child.id}", positions, tension=0.5, line_width=5.0, color=np.array([0, 0, 0]), segments=1)
            visualize_tree_helper(child, positions[1], depth + 1, len(node.children))
            
    visualize_tree_helper(root, init_start_offset, 0, 0)
    server.add_gaussian_splats(f"/gaussians_scene", centers=running_centers, rgbs=running_rgbs, opacities=running_opacities, covariances=running_covariances)

def visualize_saved_tree(pickle_file):
    with open(pickle_file, 'rb') as file:
        tree_data = pickle.load(file)
    scales = tree_data["scales"]
    wxyzs = tree_data["wxyzs"]
    means = tree_data["means"]
    opacities = tree_data["opacities"]
    colors = tree_data["rgbs"]
    root = tree_data["root"]
    visualize_tree(scales, wxyzs, means, colors, opacities, root)
    
def save_tree(root, scales, wxyzs, rgbs, means, opacities, file_path):
    # Bundle all the data into a dictionary
    tree_data = {
        "root": root,
        "scales": scales,
        "wxyzs": wxyzs,
        "rgbs": rgbs,
        "means": means,
        "opacities": opacities
    }
    
    # Serialize the data to a file
    with open(file_path, 'wb') as file:
        pickle.dump(tree_data, file)

    print(f"Tree and parameters saved to {file_path}.")
    
    
if __name__ == "__main__":
    # Control C to stop the program
    import time
    pickle_file = "./saved_tree.pkl"
    visualize_saved_tree(pickle_file)
    time.sleep(10000000) #TODO fix this hack, need to properly call viser server in the background rather than just leave python interpreter open

#TODO do a bfs algorithm, run all the masks first at each level, then sort by number of gaussians and if there is any overlap then reject those, cluster within the child node
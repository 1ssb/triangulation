import torch
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import stats
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Triangulation:
    def __init__(self, H, W, intrinsic_matrix, c2w_matrices, bounding_boxes, device=None):
        self.H = H
        self.W = W
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.intrinsic_matrix = intrinsic_matrix.to(self.device)
        self.c2w_matrices = [c2w.to(self.device) for c2w in c2w_matrices]
        self.bounding_boxes = bounding_boxes
        self.rays = {i: [] for i in range(len(c2w_matrices))}
        self.points = []
        self.sample_density = 2  # Generate rays at every 2 pixels
        logging.info("Initialized Triangulation class.")

    def generate_rays(self):
        logging.info("Generating rays...")
        for index, (c2w_matrix, bbox) in enumerate(zip(self.c2w_matrices, self.bounding_boxes)):
            cam_center = c2w_matrix[:3, 3]
            cam_orientation = c2w_matrix[:3, :3]
            xmin, ymin, xmax, ymax = bbox
            for i in range(max(xmin, 0), min(xmax, self.W), self.sample_density):
                for j in range(max(ymin, 0), min(ymax, self.H), self.sample_density):
                    x = (i - self.intrinsic_matrix[0, 2]) / self.intrinsic_matrix[0, 0]
                    y = (j - self.intrinsic_matrix[1, 2]) / self.intrinsic_matrix[1, 1]
                    pixel_coords = torch.tensor([x, y, 1], dtype=torch.float32, device=self.device)
                    ray_dir = cam_orientation @ pixel_coords
                    norm = torch.norm(ray_dir)
                    if norm > 1e-6:
                        ray_dir /= norm
                        self.rays[index].append((cam_center, ray_dir))
        logging.info("Rays generated successfully.")

    def triangulate_points(self, num_samples=1000):
        logging.info("Triangulating points...")
        if len(self.rays) < 2:
            raise ValueError("At least two sets of rays are needed for triangulation.")
        
        camera_pairs = list(combinations(self.rays.keys(), 2))
        for _ in tqdm(range(num_samples)):
            cam1, cam2 = random.choice(camera_pairs)
            r1 = random.choice(self.rays[cam1])
            r2 = random.choice(self.rays[cam2])
            origin1, dir1 = r1
            origin2, dir2 = r2
            a = torch.cross(dir1, dir2, dim=0)
            b = origin2 - origin1
            a_dot_a = torch.dot(a, a)
            if a_dot_a > 1e-6:
                t = torch.dot(torch.cross(b, dir2, dim=0), a) / a_dot_a
                point = origin1 + t * dir1
                if torch.dot(point - origin1, dir1) > 0 and torch.dot(point - origin2, dir2) > 0:
                    self.points.append(point)
        logging.info("Triangulation completed with {} points.".format(len(self.points)))

    def optimize_points(self, num_iterations=100, lr=0.01):
        logging.info("Optimizing points...")
        if not self.points:
            logging.error("No points to optimize.")
            return None
        
        points_tensor = torch.stack(self.points).to(self.device)
        points_tensor.requires_grad = True
        optimizer = torch.optim.Adam([points_tensor], lr=lr)

        for _ in tqdm(range(num_iterations)):
            optimizer.zero_grad()
            loss = 0

            for index, c2w_matrix in enumerate(self.c2w_matrices):
                cam_center = c2w_matrix[:3, 3]
                cam_orientation = c2w_matrix[:3, :3]

                reprojected_points = cam_orientation.T @ (points_tensor.T - cam_center.view(-1, 1))
                norm_reprojected_points = reprojected_points / (reprojected_points[2, :] + 1e-6)

                x = self.intrinsic_matrix[0, 0] * norm_reprojected_points[0, :] + self.intrinsic_matrix[0, 2]
                y = self.intrinsic_matrix[1, 1] * norm_reprojected_points[1, :] + self.intrinsic_matrix[1, 2]

                observed_x = (self.bounding_boxes[index][0] + self.bounding_boxes[index][2]) / 2
                observed_y = (self.bounding_boxes[index][1] + self.bounding_boxes[index][3]) / 2

                loss += torch.sum((x - observed_x) ** 2 + (y - observed_y) ** 2)
                # logging.info(loss)
            loss.backward()
            optimizer.step()

        logging.info("Optimization finished.")
        return points_tensor.detach().cpu().numpy()

    def calculate_depth_statistics(self, optimized_points):
        logging.info("Calculating depth statistics...")
        if optimized_points is None or optimized_points.shape[1] < 3:
            raise ValueError("Optimized points are not in the correct format.")
        
        depths_array = optimized_points[:, 2]
        mean_depth = np.mean(depths_array)
        mode_depth = stats.mode(depths_array).mode
        logging.info("Depth statistics calculated: Mean depth {}, Mode depth {}".format(mean_depth, mode_depth))
        return mean_depth, mode_depth, depths_array

    def plot_depth_distribution(self, depths_array, mean_depth):
        if depths_array is None:
            raise ValueError("Depth array is None, cannot plot distribution.")
        
        plt.figure(figsize=(10, 6))
        plt.hist(depths_array, bins=50, color='blue', alpha=0.7)
        plt.title('Depth Distribution')
        plt.xlabel('Depth')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.axvline(mean_depth, color='red', linestyle='dashed', linewidth=2)
        plt.show()
        logging.info("Depth distribution plotted.")

def ray_triangulator(focus, c2w_matrices, bboxes):
    # Define camera parameters and transformation matrices
    H, W = 360, 640  # Image dimensions
    intrinsic_matrix = torch.tensor([
        [focus, 0, W / 2],  # fx, 0, cx
        [0, focus, H / 2],   # 0, fy, cy
        [0, 0, 1]                # Standard row for homogeneous coordinates
    ], dtype=torch.float32)

    # Create Triangulation instance
    triangulate = Triangulation(H, W, intrinsic_matrix, c2w_matrices, bboxes)

    # Generate rays
    triangulate.generate_rays()

    # Triangulate points
    triangulate.triangulate_points()

    # Optimize points
    optimized_points = triangulate.optimize_points()

    if optimized_points is None:
        logging.error("Optimization failed. Exiting.")
        return

    # Calculate depth statistics
    mean_depth, mode_depth, depths_array = triangulate.calculate_depth_statistics(optimized_points)

    logging.info("Mean depth: {}, Mode depth: {}".format(mean_depth, mode_depth))

    # Find the 3D points closest to the mean, median, and mode depths
    mean_point = optimized_points[np.argmin(np.abs(depths_array - mean_depth))]
    mode_point = optimized_points[np.argmin(np.abs(depths_array - mode_depth))]

    logging.info("3D point closest to mean depth: {}".format(mean_point))
    logging.info("3D point closest to mode depth: {}".format(mode_point))

    # Plot depth distribution
    triangulate.plot_depth_distribution(depths_array, mean_depth)
    
if __name__ == "__main__":
    
    FOCUS = 311.368
    
    c2w_matrices = [
        torch.tensor([
            [0.99502106, 0.00362083, -0.09959916, 0.04915240],
            [-0.00466050, 0.99993702, -0.01020783, -0.04349710],
            [0.09955593, 0.01062119, 0.99497529, 0.67210966],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=torch.float32),
        torch.tensor([
            [0.99546941, 0.00413685, -0.09499240, 0.05023769],
            [-0.00512334, 0.99993541, -0.01014341, -0.04379150],
            [0.09494431, 0.01058413, 0.99542634, 0.65688956],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=torch.float32),
        torch.tensor([
            [0.98052531, 0.00387463, -0.19635468, 0.00978095],
            [-0.00610145, 0.99992371, -0.01073717, -0.05595012],
            [0.19629811, 0.01172611, 0.98047412, 0.95481133],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=torch.float32)
    ] # Warning: order matters make sure its in the same order as the bounding boxes

    bboxes = [
        (457, 170, 613, 349),
        (448, 169, 607, 357),
        (534, 183, 639, 360)
    ]
    
    ray_triangulator(FOCUS, c2w_matrices, bboxes)

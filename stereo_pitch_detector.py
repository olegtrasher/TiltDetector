import cv2
import numpy as np
import matplotlib.pyplot as plt

class StereoPitchDetector:
    def __init__(self, baseline_mm=68, fov_deg=180, stereo_size=(2048, 1024), max_points=5000):
        self.baseline = baseline_mm / 1000.0  # в метрах
        self.fov = np.radians(fov_deg)
        self.stereo_size = stereo_size
        self.max_points = max_points

    def resize_for_stereo(self, img):
        # Ресайзим до stereo_size (по ширине и высоте)
        return cv2.resize(img, self.stereo_size)

    def split_sbs(self, img):
        h, w = img.shape[:2]
        left = img[:, :w//2]
        right = img[:, w//2:]
        return left, right

    def compute_disparity(self, left, right):
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16*8,  # 128, должно делиться на 16
            blockSize=5,
            P1=8*3*5**2,
            P2=32*3*5**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        return disparity

    def disparity_to_pointcloud(self, disparity, left_img):
        h, w = disparity.shape
        f = (w) / (2 * np.tan(self.fov/2))
        Q = np.float32([[1, 0, 0, -w/2],
                        [0, -1, 0, h/2],
                        [0, 0, 0, f],
                        [0, 0, 1/self.baseline, 0]])
        points_3D = cv2.reprojectImageTo3D(disparity, Q)
        mask = (disparity > 0) & np.isfinite(disparity)
        points = points_3D[mask]
        colors = left_img[mask]
        # Ограничиваем количество точек для анализа и визуализации
        if points.shape[0] > self.max_points:
            idx = np.random.choice(points.shape[0], self.max_points, replace=False)
            points = points[idx]
            colors = colors[idx]
        print("points_3D shape:", points_3D.shape)
        print("mask shape:", mask.shape)
        print("points shape (limited):", points.shape)
        return points, colors

    def fit_plane(self, points):
        # Корректно: SVD по (N, 3), где N - число точек
        centroid = np.mean(points, axis=0)
        # SVD по центруированным точкам
        _, _, vh = np.linalg.svd(points - centroid)
        normal = vh[2, :]
        return normal, centroid

    def estimate_pitch(self, normal):
        pitch_rad = np.arcsin(normal[1])
        pitch_deg = np.degrees(pitch_rad)
        return pitch_deg

    def visualize_pointcloud(self, points, colors, normal=None, centroid=None):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:,0], points[:,1], points[:,2], c=colors/255.0, s=0.5)
        if normal is not None and centroid is not None:
            ax.quiver(centroid[0], centroid[1], centroid[2],
                      normal[0], normal[1], normal[2], length=1, color='r')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show() 
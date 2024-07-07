import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import os
import sys
from stl import mesh
from io import StringIO
import math

def detect_shapes(image):
    if image is None:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    shapes = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        num_vertices = len(approx)
        shape_name = None
        dimensions = None
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        aspect_ratio = None

        if num_vertices == 3:
            shape_name = "Triangle"
            x, y, w, h = cv2.boundingRect(approx)
            base = round(w)
            height = round(h)
            slant_height = round(np.linalg.norm(approx[0][0] - approx[1][0]))
            dimensions = (base, height, slant_height)
        elif num_vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                shape_name = "Square"
                side_length = round(w)
                dimensions = (side_length,)
            else:
                shape_name = "Rectangle"
                dimensions = (round(w), round(h))
        elif num_vertices > 4:
            if circularity > 0.8 and area > 0:
                shape_name = "Circle"
                radius = round(np.sqrt(area / np.pi))
                dimensions = (radius,)
            elif circularity < 0.6:
                shape_name = "Curved Surface (Rectangle)"
                x, y, w, h = cv2.boundingRect(approx)
                dimensions = (round(w), round(h))
            else:
                x, y, w, h = cv2.boundingRect(approx)
                shape_name = "Curved Surface (Cone)"
                dimensions = (round(w), round(h))

        if shape_name is not None and dimensions is not None:
            shapes.append((shape_name, contour, dimensions))

    return shapes

class ShapeDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shape Detector")

        self.left_frame = tk.Frame(self.root, width=400)
        self.right_frame = tk.Frame(self.root, width=400)

        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.upload_button = tk.Button(self.left_frame, text="Upload Images", command=self.upload_images, width=20, height=2, font=("Arial", 14))
        self.upload_button.pack(pady=10, anchor=tk.CENTER)

        self.run_button = tk.Button(self.left_frame, text="Run", command=self.run_detection, width=20, height=2, font=("Arial", 14))
        self.run_button.pack(pady=10, anchor=tk.CENTER)

        self.message_box = tk.Text(self.right_frame, state='disabled', width=30, font=("Arial", 14))
        self.message_box.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.file_paths = []
        self.file_counter = {"cube": 1, "sphere": 1, "cylinder": 1, "pyramid": 1, "cone": 1}

    def upload_images(self):
        self.file_paths = filedialog.askopenfilenames()
        self.log_message(f"Uploaded {len(self.file_paths)} images.")

    def run_detection(self):
        if not self.file_paths:
            self.log_message("No images uploaded. Please upload images first.")
            return

        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        shapes_detected = []

        for file_path in self.file_paths:
            image = cv2.imread(file_path)
            if image is None:
                self.log_message(f"Failed to read the image: {file_path}. Skipping this file.")
                continue

            shapes = detect_shapes(image)
            if shapes:
                shapes_detected.extend(shapes)
                self.log_message(f"Detected {len(shapes)} shape(s) in {file_path}.")
            else:
                self.log_message(f"No shapes detected in {file_path}.")

        total_shapes_detected = len(shapes_detected)
        if total_shapes_detected > 3:
            self.log_message("Complex image:- contains multiple shapes hence, cannot construct the 3d model.")
            return

        self.log_message(f"Total shapes detected: {total_shapes_detected}")
        self.construct_3d_model(shapes_detected)

        sys.stdout = old_stdout
        self.log_message(mystdout.getvalue())

    def log_message(self, message):
        self.message_box.config(state='normal')
        self.message_box.insert(tk.END, message + '\n')
        self.message_box.config(state='disabled')

    def construct_3d_model(self, shapes_detected):
        shape_count = {"Square": 0, "Rectangle": 0, "Circle": 0, "Triangle": 0, "Curved Surface (Rectangle)": 0, "Curved Surface (Cone)": 0}
        dimensions = {"Square": [], "Rectangle": [], "Circle": [], "Triangle": [], "Curved Surface (Rectangle)": [], "Curved Surface (Cone)": []}

        for shape, _, dim in shapes_detected:
            shape_count[shape] += 1
            dimensions[shape].append(dim)

        combined_mesh = None
        shape_name = None

        if shape_count["Square"] == 3:
            avg_side_length = round(np.mean([d[0] for d in dimensions["Square"]]))
            combined_mesh = self.construct_cube(avg_side_length)
            shape_name = "cube"
            surface_area = round(6 * avg_side_length**2)
            volume = round(avg_side_length**3)
            print(f"Cube dimensions: side_length={avg_side_length}, surface_area={surface_area}, volume={volume}")
        elif shape_count["Circle"] == 3:
            avg_radius = round(np.mean([d[0] for d in dimensions["Circle"]]))
            combined_mesh = self.construct_sphere(avg_radius)
            shape_name = "sphere"
            surface_area = round(4 * np.pi * avg_radius**2)
            volume = round((4/3) * np.pi * avg_radius**3)
            print(f"Sphere dimensions: radius={avg_radius}, surface_area={surface_area}, volume={volume}")
        elif shape_count["Circle"] == 2 and shape_count["Curved Surface (Rectangle)"] == 1:
            avg_radius = round(np.mean([d[0] for d in dimensions["Circle"]]))
            height = dimensions["Curved Surface (Rectangle)"][0][1]
            combined_mesh = self.construct_cylinder(avg_radius, height)
            shape_name = "cylinder"
            surface_area = round(2 * np.pi * avg_radius * (avg_radius + height))
            volume = round(np.pi * avg_radius**2 * height)
            print(f"Cylinder dimensions: radius={avg_radius}, height={height}, surface_area={surface_area}, volume={volume}")
        elif shape_count["Triangle"] == 2 and (shape_count["Rectangle"] == 1 or shape_count["Square"] == 1):
            base = round(np.mean([d[0] for d in dimensions["Triangle"]]))
            height = round(np.mean([d[1] for d in dimensions["Triangle"]]))
            if shape_count["Rectangle"] == 1:
                length = dimensions["Rectangle"][0][0]
                breadth = dimensions["Rectangle"][0][1]
                base_area = round(length * breadth)
                surface_area = round(base_area + 2 * (length * height) + 2 * (breadth * height))
            else:
                side_length = dimensions["Square"][0][0]
                base_area = round(side_length**2)
                surface_area = round(base_area + 2 * side_length * height)
            volume = round((1/3) * base_area * height)
            combined_mesh = self.construct_pyramid(base, height)
            shape_name = "pyramid"
            print(f"Pyramid dimensions: base={base}, height={height}, surface_area={surface_area}, volume={volume}")
        elif shape_count["Circle"] == 1 and shape_count["Triangle"] == 1 and shape_count["Curved Surface (Cone)"] == 1:
            radius = dimensions["Circle"][0][0]
            height = dimensions["Curved Surface (Cone)"][0][1]
            slant_height = round(np.sqrt(radius**2 + height**2))
            combined_mesh = self.construct_cone(radius, height)
            shape_name = "cone"
            surface_area = round(np.pi * radius * (radius + slant_height))
            volume = round((1/3) * np.pi * radius**2 * height)
            print(f"Cone dimensions: radius={radius}, height={height}, slant_height={slant_height}, surface_area={surface_area}, volume={volume}")
        else:
            print("No valid 3D shape could be constructed from the detected shapes.")
            return

        if combined_mesh is not None:
            self.save_mesh_to_file(combined_mesh, shape_name)

    def construct_cube(self, side_length):
        vertices = np.array([[-0.5, -0.5, -0.5],
                             [ 0.5, -0.5, -0.5],
                             [ 0.5,  0.5, -0.5],
                             [-0.5,  0.5, -0.5],
                             [-0.5, -0.5,  0.5],
                             [ 0.5, -0.5,  0.5],
                             [ 0.5,  0.5,  0.5],
                             [-0.5,  0.5,  0.5]])

        faces = np.array([[0, 3, 1], [1, 3, 2],
                          [0, 4, 7], [0, 7, 3],
                          [4, 5, 6], [4, 6, 7],
                          [5, 1, 2], [5, 2, 6],
                          [0, 1, 5], [0, 5, 4],  # Added the missing face
                          [2, 3, 7], [2, 7, 6]])

        cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                cube.vectors[i][j] = vertices[f[j], :] * side_length

        return cube

    def construct_sphere(self, radius, stack_count=18, sector_count=36):
        vertices = []
        faces = []

        for i in range(stack_count + 1):
            stack_angle = np.pi / 2 - i * np.pi / stack_count
            xy = radius * np.cos(stack_angle)
            z = radius * np.sin(stack_angle)

            for j in range(sector_count + 1):
                sector_angle = j * 2 * np.pi / sector_count
                x = xy * np.cos(sector_angle)
                y = xy * np.sin(sector_angle)
                vertices.append([round(x), round(y), round(z)])

        for i in range(stack_count):
            for j in range(sector_count):
                first = (i * (sector_count + 1)) + j
                second = first + sector_count + 1

                faces.append([first, second, first + 1])
                faces.append([second, second + 1, first + 1])

        sphere = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                sphere.vectors[i][j] = vertices[f[j]]

        return sphere

    def construct_cylinder(self, radius, height, sector_count=36):
        vertices = []
        faces = []

        for i in range(2):
            z = -height / 2.0 if i == 0 else height / 2.0
            for j in range(sector_count):
                sector_angle = j * 2 * np.pi / sector_count
                x = radius * np.cos(sector_angle)
                y = radius * np.sin(sector_angle)
                vertices.append([round(x), round(y), round(z)])

        for j in range(sector_count):
            top1 = j
            top2 = (j + 1) % sector_count
            bottom1 = j + sector_count
            bottom2 = (j + 1) % sector_count + sector_count

            faces.append([top1, bottom1, bottom2])
            faces.append([top1, bottom2, top2])

        cylinder = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                cylinder.vectors[i][j] = vertices[f[j]]

        return cylinder

    def construct_pyramid(self, base, height):
        vertices = np.array([[0, 0, 0],
                             [base, 0, 0],
                             [base, base, 0],
                             [0, base, 0],
                             [base / 2, base / 2, height]])

        faces = np.array([[0, 1, 2], [0, 2, 3],
                          [0, 1, 4], [1, 2, 4],
                          [2, 3, 4], [3, 0, 4]])

        pyramid = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                pyramid.vectors[i][j] = vertices[f[j], :]

        return pyramid

    def construct_cone(self, radius, height, sector_count=36):
        vertices = [[0, 0, 0]]
        faces = []

        for j in range(sector_count):
            sector_angle = j * 2 * np.pi / sector_count
            x = radius * np.cos(sector_angle)
            y = radius * np.sin(sector_angle)
            vertices.append([round(x), round(y), 0])

        vertices.append([0, 0, height])

        for j in range(1, sector_count + 1):
            faces.append([0, j, (j % sector_count) + 1])

        for j in range(1, sector_count + 1):
            faces.append([j, len(vertices) - 1, (j % sector_count) + 1])

        cone = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                cone.vectors[i][j] = vertices[f[j]]

        return cone

    def save_mesh_to_file(self, combined_mesh, shape_name):
        output_dir = "output_shapes"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f"{shape_name}_{self.file_counter[shape_name]}.stl")
        combined_mesh.save(output_file)
        self.log_message(f"Saved {shape_name.capitalize()} model to {output_file}")
        self.file_counter[shape_name] += 1

if __name__ == "__main__":
    root = tk.Tk()
    app = ShapeDetectorApp(root)
    root.mainloop()

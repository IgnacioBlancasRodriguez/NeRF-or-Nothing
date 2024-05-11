import json
import numpy as np

def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return image_id1, image_id2

def load_points(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or len(line.strip()) == 0:
                continue  
            data = line.strip().split()
            point_id = int(data[0])
            x, y, z = float(data[1]), float(data[2]), float(data[3])
            r, g, b = int(data[4]), int(data[5]), int(data[6])
            error = float(data[7])
            track = [(int(data[i]), int(data[i + 1])) for i in range(8, len(data), 2)]
            points.append({'point_id': point_id, 'x': x, 'y': y, 'z': z,
                           'r': r, 'g': g, 'b': b, 'error': error, 'track': track})
    return points

def normalize_coordinates(points):
    #find bounding box, scale to min and max
    min_coords = np.min([[point['x'], point['y'], point['z']] for point in points], axis=0)
    max_coords = np.max([[point['x'], point['y'], point['z']] for point in points], axis=0)
    
    scaling_factors = 2.0 / (max_coords - min_coords)
    
    #apply scaling and centering
    normalized_points = []
    for point in points:
        normalized_x = (point['x'] - min_coords[0]) * scaling_factors[0] - 1.0
        normalized_y = (point['y'] - min_coords[1]) * scaling_factors[1] - 1.0
        normalized_z = (point['z'] - min_coords[2]) * scaling_factors[2] - 1.0
        normalized_points.append({
            'point_id': point['point_id'],
            'x': normalized_x,
            'y': normalized_y,
            'z': normalized_z,
            'r': point['r'],
            'g': point['g'],
            'b': point['b'],
            'error': point['error'],
            'track': point['track']
        })
    return normalized_points

def load_image_transforms(transforms_path):
    images = []
    data = json.loads(transforms_path)

    
    return images


def generate_training_data(images_path, transforms_path, points_path):
    points = load_points(points_path)
    norm_points = normalize_coordinates(points)
    cameras = load_image_transforms(transforms_path)
    data = []
    for point in points:
        coords = [point['x'],point['x'],point['x']]
        

    return data
    

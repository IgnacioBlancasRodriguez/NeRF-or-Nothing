import json
import numpy as np
import pickle
from PIL import Image
import os
import random
from typing import List, Mapping, Optional, Text, Tuple, Union

import numpy as np
import scipy

def get_cam_transforms(json_path):
    transforms = {}
    with open(json_path, 'r') as f:
        transforms_data = f.read()

    data = json.loads(transforms_data)

    focal = data["fl_x"]
    # print('FRAMES',data["frames"])
    for frame in data["frames"]:
        matrix = frame["transform_matrix"]
        file_name = os.path.basename(frame["file_path"])
        transforms[file_name] = np.array(matrix)

    return transforms, focal

def normalize(vec):
    """Normalize a vector."""
    return vec / np.linalg.norm(vec, axis=-1, keepdims=True)


def get_ray_origins_and_directions(image_height, image_width, focal_length, camera_transform):
    #pixel grid
    u, v = np.meshgrid(np.linspace(0, image_width - 1, image_width),
                       np.linspace(0, image_height - 1, image_height))
    
    #pixel -> camera coords
    pixel_coords = np.stack([u, v, np.ones_like(u)], axis=-1)  # Shape: (image_height, image_width, 3)
    pixel_coords_flat = pixel_coords.reshape((-1, 3))  # Flatten to (num_pixels, 3)

    pixel_coords_homogeneous = np.hstack([pixel_coords_flat, np.ones((pixel_coords_flat.shape[0], 1))])
    camera_coords = np.matmul(np.linalg.inv(camera_transform), pixel_coords_homogeneous.T)
    
    #normalize camera coordinates and apply focal length
    ray_directions = camera_coords[:3, :] * focal_length
    ray_directions /= np.linalg.norm(ray_directions, axis=0)
    ray_directions = np.transpose(ray_directions)
    
    #ray origins
    camera_position = np.linalg.inv(camera_transform)[:3, 3]
    ray_origins = np.tile(camera_position[:, np.newaxis], (1, image_height * image_width)).T
    
    return ray_origins, ray_directions


def load_rgb_from_png(png_file):
    """Load flattened RGB values from a PNG file."""
    image = Image.open(png_file)
    image_rgb = image.convert('RGB')
    rgb_values = np.array(image_rgb)
    rgb_values_flat = rgb_values.reshape(-1, 3)
    return rgb_values_flat


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    ndc_o = np.zeros((H*W,3))
    ndc_d = np.zeros((H*W,3))
    for i in range(H*W):
        # o = rays_o[x][y] + rays_d[x][y]*(-1.0*(near+rays_o[x][y][2])/rays_d[x][y][2])
        o = rays_o[i] 
        ndc_o[i][0] = -focal * o[0] / (W * 0.5 * o[2])
        ndc_o[i][1] = -focal * o[1] / (H * 0.5 * o[2])
        ndc_o[i][2] =  1 + (2.0*near)/o[2]

        ndc_d[i][0] = -focal / (W * 0.5) * (rays_d[i][0]/rays_d[i][2] - o[0]/o[2])
        ndc_d[i][1] = -focal / (H * 0.5) * (rays_d[i][1]/rays_d[i][2] - o[1]/o[2])
        ndc_d[i][2] =  -2.0 * near / o[2]
    
    return ndc_o, ndc_d

def crop_and_scale_image(image_path, output_dir):
    img = Image.open(image_path)
    width, height = img.size

    min_dimension = min(width, height)

    left = (width - min_dimension) // 2
    top = (height - min_dimension) // 2
    right = (width + min_dimension) // 2
    bottom = (height + min_dimension) // 2
    img_cropped = img.crop((left, top, right, bottom))

    img_scaled = img_cropped.resize((400, 400), Image.ANTIALIAS)
    img_scaled.save(output_dir,format='PNG')

def txt_to_pkl(txt_file, training_file, testing_file):
    training_data = []
    testing_data = []
    with open(txt_file, 'r') as f:
        for line in f:
            row = [float(x) for x in line.split()]
            r = random.random()
            if r<0.8:
                training_data.append(row)
            else:
                testing_data.append(row)

    training_data = np.asarray(training_data).astype(np.float16)
    testing_data = np.asarray(testing_data).astype(np.float16)
    
    with open(training_file, 'wb') as f:
        pickle.dump(training_data, f)

    with open(testing_file, 'wb') as f:
        pickle.dump(testing_data, f)

def generate_data(transforms_dir,start_idx,end_idx):
    # focal_length = 2400
    near = 1
    height = 400
    width = 400
    og_prefix = "IMG_"
    og_file_type = ".JPG"
    scaled_dir = "datasets/fern/scaled/"
    scaled_prefix = "scaled_"
    scaled_file_type = ".png"

    transforms, focal_length = get_cam_transforms("datasets/fern/transforms.json")
    focal_length = 0.024
    data = []
    for idx in range(start_idx,end_idx):
        mat = transforms[og_prefix+str(idx)+og_file_type]
        rays_o,rays_d = get_ray_origins_and_directions(height,width,focal_length,mat)
        ndc_o, ndc_d = ndc_rays(height,width,focal_length,near,rays_o,rays_d)
        rgb = load_rgb_from_png(scaled_dir+scaled_prefix+str(idx)+scaled_file_type)

        for i in range(width*height):
            row = [ndc_o[i][0],ndc_o[i][1],ndc_o[i][2],
                    ndc_d[i][0],ndc_d[i][1],ndc_d[i][2],
                    rgb[i][0]/255.0,rgb[i][1]/255.0,rgb[i][2]/255.0]
            data.append(row)
        

        print((idx-start_idx)/(end_idx-start_idx)*100.0)


    data = np.asarray(data)
    np.savetxt("fern_data.txt",data,fmt='%f')
    return data

def load_poses(file_path):
    data = np.load(file_path)
    return data
    
# load_poses("poses_bounds.npy")

generate_data("datasets/fern/transforms.json",4026,4046)
txt_to_pkl("fern_data.txt","fern_train1.pkl","fern_test1.pkl")


    
# get_cam_transforms("datasets/person-hall/transforms.json")

# for i in range(4026,4046):
    # crop_and_scale_image('datasets/fern/images/IMG_'+str(i)+'.JPG','datasets/fern/scaled/scaled_'+str(i)+'.png')


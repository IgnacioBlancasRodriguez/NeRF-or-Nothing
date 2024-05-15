import json
import numpy as np

## ============= THIS FILE IS OBSOLETE: use data_gen.py ====================== ##

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

def normalize_coordinates(points, transforms_path):
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

def create_id_dict(images_path,image_directory_prefix):
    dict = {}
    
    with open(images_path, 'r') as file:
        line_num = 0
        for line in file:
            if line.startswith('#') or len(line.strip()) == 0:
                continue  
            if line_num % 2 == 1:
                line_num+=1
                continue
            data = line.strip().split()
            path = image_directory_prefix + str(data[9])
            dict[path] = data[0]
            line_num += 1

    return dict

def load_image_transforms(transforms_path, filepath_to_img_id):
    images = {}
    with open(transforms_path, 'r') as f:
        transforms_data = f.read()

    data = json.loads(transforms_data)
    print('FRAMES',data["frames"])
    for frame in data["frames"]:
        id = filepath_to_img_id[frame["file_path"]] #get IMAGE ID from file path name

        #get camera origin from last column of transform matrix
        camera_origin = [frame["transform_matrix"][0][3],frame["transform_matrix"][1][3],frame["transform_matrix"][2][3]]

        #set dictionary appropriately
        images[id] = camera_origin

    return images

#from colmap and colmap2nerf, you should input file paths to:
    # images.txt
    # transforms.json
    # points3D.txt
def generate_training_data(images_path, transforms_path, points_path):
    n = 0.1 #near plane
    f = 3.0 #far plane
    with open(transforms_path, 'r') as f:
        transforms_data = f.read()
    transforms = json.loads(transforms_data) #read in transforms json

    fl  = transforms["fl_x"] #focal length
    
    a = [-fl/transforms["w"]*2,-fl/transforms["h"]*2,1] #a for normalizing
    b_z = 2*n #b for normalizing
    
    
    filepath_to_img_id = create_id_dict(images_path,"./images/") #dict to go from filepath of image to image id
    points = load_points(points_path)
    transforms = load_image_transforms(transforms_path,filepath_to_img_id)
    print('IMAGES',transforms)
    data = []
    num_points = len(points)
    count = 0
    for point in points:
        coords = [point['x'],point['y'],point['z']] 
        rgb = [point['r'],point['g'],point['b']]
        for track in point["track"]:
            o = transforms[str(track[0])]
            o_prime = [a[0] * o[0] / o[2],a[1] * o[1] / o[2],a[2]+b_z/o[2]]

            direction = np.asarray(coords) - np.asarray(o)
            t = np.linalg.norm(direction)

            d = direction / t
            d_prime = [a[0]*(d[0]/d[2] - o[0]/o[2]),
                               a[1]*(d[1]/d[2] - o[1]/o[2]),
                               -b_z/o[2]]
            
            data.append([o_prime[0],o_prime[1],o_prime[2],
                         d_prime[0],d_prime[1],d_prime[2],
                        rgb[0],rgb[1],rgb[2]])
        if(count%1000==0):
            pct = (1.0*count)/num_points*100.0
            print(pct,'%')
        count+=1
        
    data = np.asarray(data)
    np.savetxt("data.txt",data,fmt='%f')

    return data

# loads existing model-ready data in the specified 9 column format. 
    # data_path: path to txt file for data
def load_training_data(data_path):
    data = []
    with open(data_path, 'r') as file:
        for line in file:
            if line.startswith('#') or len(line.strip()) == 0:
                continue  
            line = line.strip().split()
            line_data = [float(x) for x in line]
            data.append(line_data)

    return data
    
generate_training_data("datasets/person-hall/colmap_text/images.txt",
                       "datasets/person-hall/transforms.json",
                       "datasets/person-hall/colmap_text/points3D.txt")
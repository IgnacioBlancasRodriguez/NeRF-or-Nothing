import json
import numpy as np
import pickle

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

# def ndc_rays(H, W, focal, near, rays_o, rays_d):
def convert_to_ndc(origins, directions, focal, w, h, near=1.):
  """Convert a set of rays to NDC coordinates."""
  # Shift ray origins to near plane
  t = -(near + origins[..., 2]) / directions[..., 2]
  origins = origins + t[..., None] * directions

  dx, dy, dz = tuple(np.moveaxis(directions, -1, 0))
  ox, oy, oz = tuple(np.moveaxis(origins, -1, 0))

  # Projection
  o0 = -((2 * focal) / w) * (ox / oz)
  o1 = -((2 * focal) / h) * (oy / oz)
  o2 = 1 + 2 * near / oz

  d0 = -((2 * focal) / w) * (dx / dz - ox / oz)
  d1 = -((2 * focal) / h) * (dy / dz - oy / oz)
  d2 = -2 * near / oz

  origins = np.stack([o0, o1, o2], -1)
  directions = np.stack([d0, d1, d2], -1)
  return origins, directions

#from colmap and colmap2nerf, you should input file paths to:
    # images.txt
    # transforms.json
    # points3D.txt
def generate_training_data(images_path, transforms_path, points_path):
    n = 0.1 #near plane
    f = 1.0 #far plane
    with open(transforms_path, 'r') as f:
        transforms_data = f.read()
    transforms = json.loads(transforms_data) #read in camera transforms

    fl  = transforms["fl_x"] #focal length
    
    a = [-fl/transforms["w"]*2,-fl/transforms["h"]*2,1] #a for normalizing
    b_z = 2*n #b for normalizing
    
    filepath_to_img_id = create_id_dict(images_path,"./images/") #dict to go from filepath of image to image id
    points = load_points(points_path) #read in 3D points
    image_transforms = load_image_transforms(transforms_path,filepath_to_img_id)
    data = []
    num_points = len(points)
    count = 0

    rays_o = []
    rays_d = []
    rgbs = []
    for point in points:
        coords = [point['x'],point['y'],point['z']] 
        rgb = [point['r'],point['g'],point['b']]
        for track in point["track"]:
            o = image_transforms[str(track[0])]
            rays_o.append(o)


            direction = np.asarray(coords) - np.asarray(o)
            t = np.linalg.norm(direction)

            d = direction / t
            rays_d.append(d)

            rgbs.append(rgb)

        if(count%1000==0):
            pct = (1.0*count)/num_points*100.0
            print(pct,'%')
        count+=1
    
    assert(len(rgbs)==len(rays_d) and len(rays_d)==len(rays_o))
    ndc_o, ndc_d = rays_o
    # ndc_o, ndc_d = convert_to_ndc(transforms["h"],transforms["w"],fl,0.01,np.asarray(rays_o),np.asarray(rays_d))
    # ndc_o, ndc_d = convert_to_ndc(np.asarray(rays_o),np.asarray(rays_d),fl/100.0,transforms["w"],transforms["h"],1.01)
    for i in range(len(ndc_o)):
        data.append([ndc_o[i][0],ndc_o[i][1],ndc_o[i][2],
                    ndc_d[i][0],ndc_d[i][1],ndc_d[i][2],
                    rgbs[i][0]/255.0,rgbs[i][1]/255.0,rgbs[i][2]/255.0])
        
    data = np.asarray(data)
    np.savetxt("data_ndc.txt",data,fmt='%f')

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

def txt_to_pkl(txt_file, training_file, testing_file):
    data = []
    with open(txt_file, 'r') as f:
        for line in f:
            row = [float(x) for x in line.split()]
            data.append(row)

    size = len(data)
    training = np.asarray(data[:int(size/5*3)])
    testing = np.asarray(data[int(size/5*3):])
    training = training.astype(np.float16)
    testing = testing.astype(np.float16)
    
    with open(training_file, 'wb') as f:
        pickle.dump(training, f)

    with open(testing_file, 'wb') as f:
        pickle.dump(testing, f)

def convert_pkl_to_txt(input_pkl_file, output_txt_file):
    # Load data from pickle file
    with open(input_pkl_file, 'rb') as f:
        data = pickle.load(f)

    # Write data to text file
    with open(output_txt_file, 'w') as f:
        for item in data:
            f.write(str(item) + '\n')

    
generate_training_data("datasets/person-hall/colmap_text/images.txt",
                       "datasets/person-hall/transforms.json",
                       "datasets/person-hall/colmap_text/points3D.txt")

# convert_pkl_to_txt("training_data.pkl","truck_training.txt")

txt_to_pkl("data_ndc.txt","person_u_training.pkl","person_u_testing.pkl")
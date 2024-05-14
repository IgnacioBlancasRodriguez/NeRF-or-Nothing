import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
#from dataset_generation import load_images, generate_training_data
import matplotlib.pyplot as plt 

device = (
    "gpu"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Helper methods
def positional_encoding(p, L) -> torch.tensor:
    out = [p]       # Adding the original position at the front of the array seems improves results
    for j in range(L):
        out.append(torch.sin(2 ** j * p))
        out.append(torch.cos(2 ** j * p))
    return torch.cat(out, dim=1)

def get_perturbed_t_values(tn, tf, n_bins, device, batch_size):
    # Unperturbed t-values
    t = torch.linspace(tn, tf, n_bins, device=device).expand(batch_size, n_bins)

    # In order to preserve the t-values within the bounds [tn, tf]
    # We first compute the mid_points within the interval, n_bins - 1 mid_points
    mid_points = (t[:,:-1] + t[:,1:]) / 2 # Midpoints between each t_i and t_i+1
    # Then, we compute two arrays, an upper_bound, and lower_bound array
    lower_bounds = torch.cat((t[:,:1], mid_points), dim=-1)     # Starting from tn to the next mid_point until right before tf
    upper_bounds = torch.cat((mid_points, t[:,-1:]), dim=-1)    # Starting from the first mid_point unitl it reaches tf
    # Now essentially, we have created n_bins different bins for each value of t,
    # where the bins for the first and last value of t are half the size as the other ones (from tn to the first mid_point and from the last midpoint to t_f)
    # making it so the points in the interior are in bins of the same sise as the original bins,
    # and the two outer bins are of half the size (encolsing the values within the desired range)
    bin_sizes = upper_bounds - lower_bounds     # Computes the bin size corresponding to each value of t (taking into account the different in size between the two outer bins and the interior bins)

    preturbation_factor = torch.rand(t.shape, device=device)
    preturbed_t = lower_bounds + bin_sizes * preturbation_factor
    
    return preturbed_t

def generate_ray_positions(o, d, t):
    # Assuming the following parameter dimentons
    # o : [batch_size, 3]
    # d : [batch_size, 3]
    # t : 
    o_expanded = o.unsqueeze(1)
    d_expanded = d.unsqueeze(1)
    t_expanded = t.unsqueeze(2)

    return (o_expanded + d_expanded * t_expanded).reshape(-1, 3)


def cummulated_transmitance(alphas):
    # We first compute the original, unshifted product of exponentials
    unshifted_prod = torch.cumprod(alphas, dim=1)
    # We then shift the values to account for the i - 1 in the summation for each T_i
    shifted_prod = torch.cat(
        (torch.ones((alphas.shape[0], 1), device=alphas.device),
        unshifted_prod[:,:-1]),
        dim = -1)
    return shifted_prod

def get_rays_total_colors(nerf_model, origins, directions, tn, tf, n_bins):
    device = origins.device
    batch_size = origins.shape[0]
    t = get_perturbed_t_values(tn, tf, n_bins, device, origins.shape[0])
    
    # Added a large factor in the end to account for the lack of one in the equation
    deltas = torch.cat((t[:,1:] - t[:,:-1],
                        torch.tensor([1e10], device=device).expand(batch_size, 1)), dim=-1)

    # Generate the n_bins positions along each ray of the batch
    x = generate_ray_positions(origins, directions, t)
    # Repeats the ray direction for each point in n_bins and mathces the directions vector shape to that of the x positions vector
    expanded_directions = directions.expand(n_bins, batch_size, 3).transpose(0, 1)
    # Fetches the colors and densities predicted by the model for each point in each ray of the batch 
    colors, densities = nerf_model(x.reshape(-1, 3),        # We reshape the positions vector to match that of the input expected by the model
        expanded_directions.reshape(-1, 3))                 # We reshape the positions vector to match that of the input expected by the model
    
    # We reshape the color and density matrices to match those as before in x ([batch_size * n_bins] -> [batch_size, n_bins])
    colors, densities = (colors.reshape(batch_size, n_bins, 3),
                         densities.reshape(batch_size, n_bins))
    
    # As described in the paper, we will compute the following alpha_values
    alphas = 1 - torch.exp(- densities * deltas)
    
    T = cummulated_transmitance(1 - alphas)

    # The weights we multiply the color with the the formula provided by the paper
    weights = T.unsqueeze(2) * alphas.unsqueeze(2)
    color_values = (weights * colors).sum(dim=1)
    # In order to counteract any effects of the weights totaling to something over one,
    # and thus get the correct background for the image, we regularize the color
    sum_weights_per_ray = weights.sum(dim=-1).sum(dim=-1)
    return color_values + (1 - sum_weights_per_ray.unsqueeze(-1))


# Define the NeRF model
class NeRFModel(nn.Module):
    def __init__(self, L_pos=10, L_dir=4, num_percep_layer=256):
        super(NeRFModel, self).__init__()
        self.pos_dim_L = L_pos
        self.dir_dim_L = L_dir

        self.base_mlp_head = nn.Sequential(
            nn.Linear(L_pos * 6 + 3, num_percep_layer), nn.ReLU(),    # Adding 3 to account for the original non-encoded vector
            nn.Linear(num_percep_layer, num_percep_layer), nn.ReLU(),
            nn.Linear(num_percep_layer, num_percep_layer), nn.ReLU(),
            nn.Linear(num_percep_layer, num_percep_layer), nn.ReLU(), )
        
        # density estimation
        self.density_estimation = nn.Sequential(
            nn.Linear(L_pos * 6 + num_percep_layer + 3, num_percep_layer), nn.ReLU(),  # Adding 3 to account for the original non-encoded vector
            nn.Linear(num_percep_layer, num_percep_layer), nn.ReLU(),
            nn.Linear(num_percep_layer, num_percep_layer), nn.ReLU(),
            nn.Linear(num_percep_layer, num_percep_layer + 1), )
        
        # color prediction
        self.color_prediction = nn.Sequential(
            nn.Linear(L_dir * 6 + num_percep_layer + 3, num_percep_layer // 2), nn.ReLU(), # Adding 3 to account for the original non-encoded vector
            nn.Linear(num_percep_layer // 2, 3), nn.Sigmoid(), )

        self.relu = nn.ReLU()

    def forward(self, x, d):
        # Implement the forward pass of your NeRF model here
        x_encoded = positional_encoding(x, self.pos_dim_L)
        d_encoded = positional_encoding(d, self.dir_dim_L)
        
        first_pass = self.base_mlp_head(x_encoded)
        
        # Second pass
        second_pass_density_inter = self.density_estimation(
            torch.cat((first_pass, x_encoded), dim=1))
        
        second_pass, density = (second_pass_density_inter[:, :-1],
                                self.relu(second_pass_density_inter[:, -1]))
        # Color estimation
        color = self.color_prediction(
            torch.cat((second_pass, d_encoded), dim=1))
        
        return color, density

def train(nerf_model, optimizer, data_loader, device, tn=0, tf=1, n_bins=192):
    nerf_model.train()
    size = len(data_loader.dataset)
    for batch_num, batch in enumerate(data_loader):
        origins = batch[:,:3].to(device)
        directions = batch[:,3:6].to(device)
        ground_truths = batch[:,6:].to(device)

        generated_ray_colors = get_rays_total_colors(nerf_model, origins, directions, tn, tf, n_bins)
        loss = ((ground_truths - generated_ray_colors)**2).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_num % 100 == 0:
            loss, current = loss.item(), (batch_num + 1) * len(batch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

@torch.no_grad
def test(model, tn, tf, dataset, batch_size=10, idx=0, n_bins=192, H=400, W=400):
    model.eval()
    ray_origins = dataset[idx * H * W: (idx + 1) * H * W, :3]
    ray_directions = dataset[idx * H * W: (idx + 1) * H * W, 3:6]

    data = []   # list of regenerated pixel values
    for i in range(int(np.ceil(H / batch_size))):   # iterate over chunks
        # Get chunk of rays
        origins = ray_origins[i * W * batch_size: (i + 1) * W * batch_size].to(device)
        directions = ray_directions[i * W * batch_size: (i + 1) * W * batch_size].to(device)        
        regenerated_px_values = get_rays_total_colors(model, origins, directions, tn=tn, tf=tf,
                                                      n_bins=n_bins)
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    plt.figure()
    plt.imshow(img)
    plt.savefig(f'img_{idx}.png', bbox_inches='tight')
    plt.close()

def perform_training(model, optimizer, scheduler, train_dataloader, test_data,
                     n_epochs, device, tn, tf, n_bins, H, W):
    epochs = n_epochs
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(model, optimizer, train_dataloader, device, tn=tn, tf=tf, n_bins=192)
        scheduler.step()
        
        for img_idx in range(4):
            test(model, tn, tf, test_data, 10, img_idx, n_bins, H, W)
        torch.save(model.state_dict(), "model.pth")
        print("Saved PyTorch Model State to model.pth")

# Perform 3D reconstruction using NeRF
def perform_NeRF(image):
    # Load the images
    training_dataset = torch.from_numpy(np.load('training_data.pkl', allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True))

    # Initialize the NeRF model
    model = NeRFModel().to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Train the NeRF model using the image data
    # Implement the training loop here
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)
    data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
    perform_training(model, model_optimizer, scheduler, data_loader, testing_dataset, n_epochs=16, device=device, tn=2, tf=6, n_bins=192, H=400,
          W=400)

    # Return the reconstructed 3D model

# Main function
def main():
    # Define the list of images
    # image_list = []
    
    # last_img_idx = 1000
    # img_base_name = "image"
    # for i in range(last_img_idx):
    #     image_list.append(img_base_name + str(i) + ".jpg")

    # Reconstruct 3D model using NeRF
    reconstructed_model = perform_NeRF([])

    # Save the reconstructed 3D model
    # Implement saving the 3D model here

if __name__ == '__main__':
    try:
        # main()

        testing_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True))
        model = NeRFModel().to(device)
        model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))

        model.eval()
        with torch.no_grad():
            for img_idx in range(200):
                test(model, 2, 6, testing_dataset, 10, img_idx, 192, 400, 400)
    except KeyboardInterrupt:
        quit()
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset_generation import load_images, generate_training_data
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
    out = []
    for i in range(L):
        out.append(torch.sin((2 ** i) * torch.pi * p))
        out.append(torch.cos((2 ** i) * torch.pi * p))
    return torch.cat(out, dim=1)

def generate_ray_t_values(tn, tf, n_bins):
    lower_bound = torch.linspace(tn, tf, n_bins)
    upper_bound = torch.cat((lower_bound[1:], torch.tensor([lower_bound[-1]])), 0)
    u = torch.rand(n_bins)
    
    return lower_bound + (upper_bound - lower_bound) * u

def generate_ray_positions(o, d, t, n_bins, batch_size):
    # Generates the ray positions
    # o : origin position of the ray
    # d : Direction of the ray
    # t : array of t values of the ray
    expanded_t = t.unsqueeze(1).expand(batch_size, n_bins, -1).to(device)
    # o : [batch_size, 3]
    # o_expanded : [batch_size, n_bins, 3]
    o_expanded = o.unsqueeze(1).expand(-1, n_bins, -1) #[batch_size, n_bins, 3] 
    d_expanded = d.unsqueeze(1).expand(-1, n_bins, -1) #[batch_size, n_bins, 3]
    return (o_expanded + d_expanded * expanded_t).reshape(-1, 3)


def cummulated_transmitance(distances, densities):
    densities_transposed = torch.cat((
        torch.zeros((densities.shape[0], )).unsqueeze(1).to(device),
        densities[:,1:]), dim=1)
    T_i = torch.exp(distances * densities_transposed)
    return 1 / torch.cumprod(T_i, dim=1)

def get_rays_total_colors(origins, directions, tn, tf, n_bins, batch_size, nerf_model):
    t = generate_ray_t_values(tn, tf, n_bins)
    expanded_t = t.expand(batch_size, n_bins).to(device)

    x = generate_ray_positions(origins, directions, t, n_bins, batch_size)
    colors, densities = nerf_model(x,
        directions.unsqueeze(1).expand(-1, n_bins, -1).reshape(-1, 3))
    colors, densities = colors.reshape(
        batch_size, n_bins, 3), densities.reshape(batch_size, n_bins)
    distances_transmitance = torch.cat((
        torch.zeros((batch_size, )).unsqueeze(1).to(device),
        expanded_t[:,1:] - expanded_t[:,:-1]), dim=1)
    
    added_perturbation = (tf * torch.ones(batch_size) + 
                          torch.rand(batch_size) * ((1/n_bins) * (tf - tn)))
    distances = torch.cat(
        (distances_transmitance[:,1:],
         (added_perturbation.unsqueeze(1).to(device) - expanded_t[:,-1].unsqueeze(1))), dim=1)
    
    T = cummulated_transmitance(distances_transmitance, densities)

    weights = T * (1 - torch.exp(-1 * densities * distances))

    return (weights.unsqueeze(2) * colors).sum(dim=1)


# Define the NeRF model
class NeRFModel(nn.Module):
    def __init__(self, L_pos=10, L_dir=4, num_percep_layer=256):
        super(NeRFModel, self).__init__()
        self.pos_dim_L = L_pos
        self.dir_dim_L = L_dir

        self.base_mlp_head = nn.Sequential(
            nn.Linear((2 * L_pos) * 3, num_percep_layer), nn.ReLU(),    # Possibly add 3 to the input size
            nn.Linear(num_percep_layer, num_percep_layer), nn.ReLU(),
            nn.Linear(num_percep_layer, num_percep_layer), nn.ReLU(),
            nn.Linear(num_percep_layer, num_percep_layer), nn.ReLU(), )
        
        # density estimation
        self.density_estimation = nn.Sequential(
            nn.Linear(L_pos * 6 + num_percep_layer , num_percep_layer), nn.ReLU(),  # Possibly add 3 to the input size
            nn.Linear(num_percep_layer, num_percep_layer), nn.ReLU(),
            nn.Linear(num_percep_layer, num_percep_layer), nn.ReLU(),
            nn.Linear(num_percep_layer, num_percep_layer + 1), )
        
        # color prediction
        self.color_prediction = nn.Sequential(
            nn.Linear(L_dir * 6 + num_percep_layer, num_percep_layer // 2), nn.ReLU(),
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
        
        second_pass, density = second_pass_density_inter[:, :-1], self .relu(second_pass_density_inter[:, -1])
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

        generated_ray_colors = get_rays_total_colors(
            origins, directions, tn, tf, n_bins, batch.shape[0], nerf_model)
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
        regenerated_px_values = get_rays_total_colors(origins, directions, tn=tn, tf=tf,
                                                      n_bins=n_bins, batch_size=W * batch_size, nerf_model=model)
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    plt.figure()
    plt.imshow(img)
    plt.savefig(f'novel_views/img_{idx}.png', bbox_inches='tight')
    plt.close()

def perform_training(model, optimizer, scheduler, train_dataloader, test_data,
                     n_epochs, device, tn, tf, n_bins, H, W):
    epochs = n_epochs
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(model, optimizer, train_dataloader, device, tn=tn, tf=tf, n_bins=192)
        scheduler.step()
        
        for img_idx in range(200):
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
        main()
    except KeyboardInterrupt:
        quit()
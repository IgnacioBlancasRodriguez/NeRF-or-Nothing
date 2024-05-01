import numpy as np
import torch
from torch import nn

device = (
    "gpu"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Helper methods
def positional_encoding(p, L) -> torch.tensor:
    out = np.zeros(2 * L)
    for i in range(L):
        out[i] = torch.sin((2 ** i) * torch.pi * p)
        out[i + 1] = torch.cos((2 ** i) * torch.pi * p)
    return out

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
        second_pass, density = self.density_estimation(
            torch.cat((first_pass, x_encoded), dim=1))
        # Color estimation
        color = self.color_prediction(
            torch.cat(second_pass, d_encoded), dim=1)
        
        return color, density

def train():
    pass

def test():
    pass

def perform_training():
    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
        torch.save(model.state_dict(), "model.pth")
        print("Saved PyTorch Model State to model.pth")

# Load the list of images
def load_images(image_list):
    # Implement image loading and preprocessing here
    return images

# Generate training data using SFM
def generate_training_data(images):
    pass
# Perform 3D reconstruction using NeRF
def reconstruct_3d(images):
    # Load the images
    image_data = load_images(images)

    # Initialize the NeRF model
    model = NeRFModel().to(device)

    # Train the NeRF model using the image data
    # Implement the training loop here

    # Perform 3D reconstruction using the trained model
    # Implement the 3D reconstruction algorithm here

    # Return the reconstructed 3D model
    return reconstructed_model

# Main function
def main():
    # Define the list of images
    image_list = []
    
    last_img_idx = 1000
    img_base_name = "image"
    for i in range(last_img_idx):
        image_list.append(img_base_name + str(i) + ".jpg")

    # Reconstruct 3D model using NeRF
    reconstructed_model = reconstruct_3d(image_list)

    # Save the reconstructed 3D model
    # Implement saving the 3D model here

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        quit()
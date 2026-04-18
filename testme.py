import torch

def gen_pixel_grid(width,height): 

    half_w= int(width/2)
    half_h= int(height/2)
    #1. Define the 1D range for each dimension
    x = torch.arange(-half_w, half_w + 1, 1) 
    x = torch.cat([x[:half_w], x[half_w + 1:]])

    y = torch.arange(-half_h, half_h + 1, 1) 
    y = torch.cat([y[:half_h], y[half_h + 1:]])

    # 2. Create meshgrid, which returns dense grids (25, 25)
    # Using indexing='ij' for Cartesian-like behavior
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    # 3. Stack and reshape to get (Number of points, 2)
    # stack gives (2, 5, 5), permute changes to (5, 5, 2), flatten to (25, 2)
    grid_points = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)

    return grid_points


print(gen_pixel_grid(4,4));
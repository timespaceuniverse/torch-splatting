import torch

def gen_pixel_grid(width,height): 

        half_w= int(width/2)
        half_h= int(height/2)
        #1. Define the 1D range for each dimension
        x = torch.arange(-half_w+0.5, half_w+0.5, 1) 
        y = torch.arange(half_h-0.5 ,-half_h-0.5,-1) 

        # 2. Create meshgrid, which returns dense grids (25, 25)
        # Using indexing='ij' for Cartesian-like behavior
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        # 3. Stack and reshape to get (Number of points, 2)
        # stack gives (2, 5, 5), permute changes to (5, 5, 2), flatten to (25, 2)
        grid_points = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)

        return grid_points.reshape((width, height, 2))

print(gen_pixel_grid(4,4)[0,1]);
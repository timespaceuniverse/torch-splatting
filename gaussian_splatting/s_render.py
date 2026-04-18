import pdb
import torch
import torch.nn as nn
import math
from einops import reduce

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def homogeneous(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

 


def projection_ndc(points, viewmatrix, projmatrix):
    points_o = homogeneous(points) # object space
    points_h = points_o @ viewmatrix @ projmatrix # screen space # RHS
    p_w = 1.0 / (points_h[..., -1:] + 0.000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    in_mask = p_view[..., 2] >= 0.2
    return p_proj, in_mask


@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max


from .utils.sh_utils import eval_sh
import torch.autograd.profiler as profiler
USE_PROFILE = False
import contextlib

class SRenderer(nn.Module):
    """
    A gaussian splatting renderer

    >>> gaussModel = GaussModel.create_from_pcd(pts)
    >>> gaussRender = GaussRenderer()
    >>> out = gaussRender(pc=gaussModel, camera=camera)
    """

    def __init__(self, active_sh_degree=3, white_bkgd=True, **kwargs):
        super(SRenderer, self).__init__()
        self.active_sh_degree = active_sh_degree
        self.debug = False
        self.white_bkgd = white_bkgd
        self.pix_coord = None
        

    
    def render(self, camera, means2D,color, opacity,scales,pixel_view_direction_world):
        #radii = get_radius(cov2d)
        #rect = get_rect(means2D, radii, width=camera.image_width, height=camera.image_height)

        if(self.pix_coord is None):
            self.pix_coord = torch.stack(torch.meshgrid(torch.arange(camera.image_width), torch.arange(camera.image_height), indexing='xy'), dim=-1).to('cuda')

        self.render_color = torch.ones(*self.pix_coord.shape[:2], 3).to('cuda')
        self.render_depth = torch.zeros(*self.pix_coord.shape[:2], 1).to('cuda')
        self.render_alpha = torch.zeros(*self.pix_coord.shape[:2], 1).to('cuda')

        TILE_SIZE = 64
        for h in range(0, camera.image_height, TILE_SIZE):
            for w in range(0, camera.image_width, TILE_SIZE):
                # check if the rectangle penetrate the tile
                over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
                over_br = rect[1][..., 0].clip(max=w+TILE_SIZE-1), rect[1][..., 1].clip(max=h+TILE_SIZE-1)
                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 
                
                if not in_mask.sum() > 0:
                    continue

                P = in_mask.sum()
                tile_coord = self.pix_coord[h:h+TILE_SIZE, w:w+TILE_SIZE].flatten(0,-2)
                sorted_depths, index = torch.sort(depths[in_mask])
                sorted_means2D = means2D[in_mask][index]
                sorted_cov2d = cov2d[in_mask][index] # P 2 2
                sorted_conic = sorted_cov2d.inverse() # inverse of variance
                sorted_opacity = opacity[in_mask][index]
                sorted_color = color[in_mask][index]
                dx = (tile_coord[:,None,:] - sorted_means2D[None,:]) # B P 2
                
                gauss_weight = torch.exp(-0.5 * (
                    dx[:, :, 0]**2 * sorted_conic[:, 0, 0] 
                    + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]
                    + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1]
                    + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0]))
                
                alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99) # B P 1
                T = torch.cat([torch.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).cumprod(dim=1)
                acc_alpha = (alpha * T).sum(dim=1)
                tile_color = (T * alpha * sorted_color[None]).sum(dim=1) + (1-acc_alpha) * (1 if self.white_bkgd else 0)
                tile_depth = ((T * alpha) * sorted_depths[None,:,None]).sum(dim=1)
                self.render_color[h:h+TILE_SIZE, w:w+TILE_SIZE] = tile_color.reshape(TILE_SIZE, TILE_SIZE, -1)
                self.render_depth[h:h+TILE_SIZE, w:w+TILE_SIZE] = tile_depth.reshape(TILE_SIZE, TILE_SIZE, -1)
                self.render_alpha[h:h+TILE_SIZE, w:w+TILE_SIZE] = acc_alpha.reshape(TILE_SIZE, TILE_SIZE, -1)

        return {
            "render": self.render_color,
            "depth": self.render_depth,
            "alpha": self.render_alpha,
            "visiility_filter": radii > 0,
            "radii": radii
        }



    def gen_pixel_grid(self,width,height): 

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


    def forward(self, camera, pc, **kwargs):
        xyz = pc.get_xyz
        color = pc.get_color
        opacity = pc.get_opacity
        scales = pc.get_scaling
        
        if USE_PROFILE:
            prof = profiler.record_function
        else:
            prof = contextlib.nullcontext
            
        with prof("projection"):
            mean_ndc, in_mask = projection_ndc(xyz, 
                    viewmatrix=camera.world_view_transform, 
                    projmatrix=camera.projection_matrix)
            assert in_mask.any(), "No points in the frustum"
            mean_ndc = mean_ndc[in_mask]

        with prof("view direction"):       
            pixel_grid=self.gen_pixel_grid(camera.image_width,camera.image_height)
            pixel2locationMatrix= torch.tensor([[1/camera.focal_x],[1/camera.focal_y]])
            pixel_xy = pixel_grid @ pixel2locationMatrix # z actuall equals 1
            pixel_xyz = torch.cat((pixel_xy, torch.ones((pixel_xy.shape[0],2))), dim=-1)
            #convert it to the world space
            pixel_xyz = pixel_xyz @ (camera.c2w.permute(1,0))
            pixel_xyz = pixel_xyz[:,:-1] #remove the last 1 
            #the pixel related view normalized direction in world space
            pixel_view_direction_world = torch.nn.functional.normalize(pixel_xyz, p=2, dim=-1)

        
        with prof("build color"):
            mean_coord_x = ((mean_ndc[..., 0] + 1) * camera.image_width - 1.0) * 0.5
            mean_coord_y = ((mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
            means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)
        
        with prof("render"):
            rets = self.render(
                camera = camera, 
                means2D=means2D,
                color=color,
                opacity=opacity, 
                scales = scales,
                pixel_view_direction_world=pixel_view_direction_world,
            )

        return rets

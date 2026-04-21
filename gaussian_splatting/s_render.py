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



@torch.no_grad()
def cal_depth(points,camera):
    
    points_o = homogeneous(points) # object space
    p_view = points_o @ camera.world_view_transform
    points_z=p_view[:,2]
    return points_z


@torch.no_grad()
def cal_pixel_boundary(points,scales,camera):
    
    points_o = homogeneous(points) # object space
    p_view = points_o @ camera.world_view_transform
    
    points_x=p_view[:,0]
    points_y=p_view[:,1]
    points_z=p_view[:,2]

    u_delta=scales*torch.sqrt(points_x*points_x+points_z*points_z-scales*scales)
    v_delta=scales*torch.sqrt(points_y*points_y+points_z*points_z-scales*scales)

    denominator = points_z*points_z-scales*scales

    u_min=camera.focal_x*(points_x*points_z-u_delta)/denominator
    u_max=camera.focal_x*(points_x*points_z+u_delta)/denominator

  
    v_min=camera.focal_y*(points_y*points_z-v_delta)/denominator
    v_max=camera.focal_y*(points_y*points_z+v_delta)/denominator

    #change to non negative pixel style
    u_min=u_min+0.5*camera.image_width
    u_max=u_max+0.5*camera.image_width
    v_min=v_min+0.5*camera.image_height
    v_max=v_max+0.5*camera.image_height

    u_min_mask= u_min < camera.image_width-1
    u_max_mask= u_max > 1
    v_min_mask= v_min < camera.image_height-1
    v_max_mask= v_max > 1

    boundary_mask= u_min_mask&u_max_mask&v_min_mask&v_max_mask

    pixel_boundary= torch.cat([
        torch.unsqueeze(u_min,dim=-1),
        torch.unsqueeze(u_max,dim=-1),
        torch.unsqueeze(v_min,dim=-1),
        torch.unsqueeze(v_max,dim=-1) ], dim=-1)
    
    return pixel_boundary[boundary_mask],boundary_mask


# def projection(points,camera):
#     viewmatrix=camera.world_view_transform, 
#     projmatrix=camera.projection_matrix #ndc projection matrix
#     points_o = homogeneous(points) # object space
#     points_h = points_o @ viewmatrix @ projmatrix # screen space   projmatrix is ndc style
#     p_w = 1.0 / (points_h[..., -1:] + 0.000001)
#     p_proj = points_h * p_w
#     p_view = points_o @ viewmatrix
#     in_mask = p_view[..., 2] >= 0.2
#     return p_proj, in_mask 

@torch.no_grad()
def cal_visible_mask(points,scales,camera):

    points_o = homogeneous(points) # object space
    p_view = points_o @ camera.world_view_transform
    #in_mask = p_view[..., 2] >= 0.    
    in_mask = p_view[..., 2]-scales >= 0.2 # a stronger requirement that takes boundary into consideration
    return in_mask


# @torch.no_grad()
# def get_rect(pix_coord, radii, width, height):
#     rect_min = (pix_coord - radii[:,None])
#     rect_max = (pix_coord + radii[:,None])
#     rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
#     rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
#     rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
#     rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
#     return rect_min, rect_max


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
        

    

    

    def render(self, camera, pixel_boundary,points,color, opacity_sigma,scales,
               depth,pixel_direction_world,pixel_grid):

        self.render_color = torch.ones(*pixel_grid.shape[:2], 3).to('cuda')
        self.render_alpha = torch.zeros(*pixel_grid.shape[:2], 1).to('cuda')
        self.render_depth = torch.zeros(*pixel_grid.shape[:2], 1).to('cuda')

        #cal the critical points of all xyz  , 1-exp(-2*scale*sigma*sqrt(1-(d/R)^2))
        critical_ratio= torch.tensor([0,0.3,0.6,0.9,1],device="cuda")
        critical_points= 1-torch.exp(torch.outer(-2*scales*opacity_sigma,torch.sqrt(1-critical_ratio)))

        TILE_SIZE = 64
        for w in range(0, camera.image_width, TILE_SIZE):
            for h in range(0, camera.image_height, TILE_SIZE):
                over_tl = pixel_boundary[:,0].clip(min=w) , pixel_boundary[:,2].clip(min=h)
                over_br = pixel_boundary[:,1].clip(max=w+TILE_SIZE-1), pixel_boundary[:,3].clip(max=h+TILE_SIZE-1)
                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # in this tile

                if not in_mask.sum() > 0:
                    continue

                sorted_depths, index = torch.sort(depth[in_mask])  
                sorted_xyz = points[in_mask][index]
                sorted_opacity_sigma = opacity_sigma[in_mask][index]
                sorted_scales = scales[in_mask][index]
                sorted_color  = color[in_mask][index]
                
                tile_pixel_direction_world = pixel_direction_world[w:w+TILE_SIZE,h:h+TILE_SIZE]

                view_ray_center_vector=torch.cross(tile_pixel_direction_world.unsqueeze(2) ,sorted_xyz.unsqueeze(0).unsqueeze(0))
                view_ray_center_dsqaure=(view_ray_center_vector**2).sum(dim=-1)
                view_ray_c_d_ratio = view_ray_center_dsqaure/(sorted_scales*sorted_scales)
                print(view_ray_c_d_ratio.shape)

                #critical points , distance => final opacity and color
                tile_alpha = torch.zeros(view_ray_c_d_ratio.shape,device="cuda")
                c01_index= view_ray_c_d_ratio <=0.3 
                tile_alpha[c01_index]= view_ray_c_d_ratio[c01_index]
                print(c01_index.shape)
                



        # if(self.pix_coord is None):
        #     self.pix_coord = torch.stack(
        #         torch.meshgrid(torch.arange(camera.image_width),
        #         torch.arange(camera.image_height), indexing='xy'), dim=-1).to('cuda')

        # self.render_color = torch.ones(*self.pix_coord.shape[:2], 3).to('cuda')
        # self.render_depth = torch.zeros(*self.pix_coord.shape[:2], 1).to('cuda')
        # self.render_alpha = torch.zeros(*self.pix_coord.shape[:2], 1).to('cuda')

        # TILE_SIZE = 64
        # for h in range(0, camera.image_height, TILE_SIZE):
        #     for w in range(0, camera.image_width, TILE_SIZE):
        #         # check if the rectangle penetrate the tile
        #         over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
        #         over_br = rect[1][..., 0].clip(max=w+TILE_SIZE-1), rect[1][..., 1].clip(max=h+TILE_SIZE-1)
        #         in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 
                
        #         if not in_mask.sum() > 0:
        #             continue

        #         P = in_mask.sum()
        #         tile_coord = self.pix_coord[h:h+TILE_SIZE, w:w+TILE_SIZE].flatten(0,-2)
        #         sorted_depths, index = torch.sort(depths[in_mask])
        #         sorted_means2D = means2D[in_mask][index]
        #         sorted_cov2d = cov2d[in_mask][index] # P 2 2
        #         sorted_conic = sorted_cov2d.inverse() # inverse of variance
        #         sorted_opacity = opacity[in_mask][index]
        #         sorted_color = color[in_mask][index]
        #         dx = (tile_coord[:,None,:] - sorted_means2D[None,:]) # B P 2
                
        #         gauss_weight = torch.exp(-0.5 * (
        #             dx[:, :, 0]**2 * sorted_conic[:, 0, 0] 
        #             + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]
        #             + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1]
        #             + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0]))
                
        #         alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99) # B P 1
        #         T = torch.cat([torch.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).cumprod(dim=1)
        #         acc_alpha = (alpha * T).sum(dim=1)
        #         tile_color = (T * alpha * sorted_color[None]).sum(dim=1) + (1-acc_alpha) * (1 if self.white_bkgd else 0)
        #         tile_depth = ((T * alpha) * sorted_depths[None,:,None]).sum(dim=1)
        #         self.render_color[h:h+TILE_SIZE, w:w+TILE_SIZE] = tile_color.reshape(TILE_SIZE, TILE_SIZE, -1)
        #         self.render_depth[h:h+TILE_SIZE, w:w+TILE_SIZE] = tile_depth.reshape(TILE_SIZE, TILE_SIZE, -1)
        #         self.render_alpha[h:h+TILE_SIZE, w:w+TILE_SIZE] = acc_alpha.reshape(TILE_SIZE, TILE_SIZE, -1)

        # return {
        #     "render": self.render_color,
        #     "depth": self.render_depth,
        #     "alpha": self.render_alpha,
        #     "visiility_filter": radii > 0,
        #     "radii": radii
        # }

        return {}


    @torch.no_grad()
    def gen_pixel_grid(self,width,height): 
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


    def forward(self, camera, pc, **kwargs):
        
        
        if USE_PROFILE:
            prof = profiler.record_function
        else:
            prof = contextlib.nullcontext

        with prof("view direction"):       
            pixel_grid=self.gen_pixel_grid(camera.image_width,camera.image_height)
            pixel_grid = pixel_grid.to(device="cuda", dtype=torch.float)
            pixel2locationMatrix= torch.tensor([1/camera.focal_x,1/camera.focal_y],device="cuda", dtype=torch.float)
            pixel_xy = pixel_grid * pixel2locationMatrix # z actuall equals 1
            pixel_xyz = torch.cat((pixel_xy, torch.ones((pixel_xy.shape[0],pixel_xy.shape[1],2),device="cuda", dtype=torch.float)), dim=-1)
            #convert it to the world space
            pixel_xyz = pixel_xyz @ (camera.c2w.permute(1,0))
            pixel_xyz = pixel_xyz[:,:,:-1] #remove the last 1 
            #cal the vector point to camer center
            pixel_xyz[:,:,0]=pixel_xyz[:,:,0]- camera.c2w[0,3]
            pixel_xyz[:,:,1]=pixel_xyz[:,:,1]- camera.c2w[1,3]
            pixel_xyz[:,:,2]=pixel_xyz[:,:,2]- camera.c2w[2,3]
            #the pixel related view normalized direction in world space
            pixel_view_direction_world = torch.nn.functional.normalize(pixel_xyz, p=2, dim=-1)
            
        # with prof("projection"):
        #     mean_ndc, in_mask = projection(xyz,camera)
        #     assert in_mask.any(), "No points in the frustum"
        #     mean_ndc = mean_ndc[in_mask]

        with prof("cal_visible_mask"):
            in_mask=cal_visible_mask(pc.get_xyz,pc.get_scaling,camera)
            assert in_mask.any(), "No points in the frustum"
            xyz = pc.get_xyz[in_mask]
            color = pc.get_color[in_mask]
            opacity_sigma = pc.get_opacity_sigma[in_mask]
            scales = pc.get_scaling[in_mask]


        with prof("cal boundary"):
            pixel_boundary,boundary_mask = cal_pixel_boundary(xyz,scales,camera)
            xyz = xyz[boundary_mask]
            color = color[boundary_mask]
            opacity_sigma = opacity_sigma[boundary_mask]
            scales = scales[boundary_mask]

        with prof("cal depth"):
            depth = cal_depth(xyz,camera)
        

        with prof("render"):
            rets = self.render(
                camera = camera, 
                pixel_boundary=pixel_boundary,
                points = xyz,
                color=color,
                opacity_sigma=opacity_sigma, 
                scales = scales,
                depth = depth,
                pixel_direction_world=pixel_view_direction_world,
                pixel_grid=pixel_grid,
            )

        return rets

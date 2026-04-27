import torch
import torch.nn  as nn
import numpy as np
import math
from simple_knn._C import distCUDA2
from gaussian_splatting.utils.point_utils import PointCloud
from gaussian_splatting.gauss_render import strip_symmetric, inverse_sigmoid, build_scaling_rotation
from gaussian_splatting.utils.sh_utils import RGB2SH

class SModel(nn.Module):
   
 
    
    def __init__(self, sh_degree : int=3, debug=False):
        super(SModel, self).__init__()
        self.max_sh_degree=sh_degree
        # self._xyz = torch.empty(0)
        # self._features_dc = torch.empty(0)
        # self._features_rest = torch.empty(0)
        # self._scaling = torch.empty(0)
        # self._rotation = torch.empty(0)
        # self._opacity = torch.empty(0)
        # self.setup_functions()
        self.debug = debug

    def create_from_pcd(self, pcd:PointCloud , camera):
        """
            create the guassian model from a color point cloud
        """
        points = pcd.coords
        colors = pcd.select_channels(['R', 'G', 'B']) / 255.

    
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        fused_color =  torch.tensor(np.asarray(colors)).float().cuda()


        

        #fused_point_cloud= fused_point_cloud[:1]
        #fused_color= fused_color[:1]

        #fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float().cuda())

        #print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        #features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        #lets try to initilize color with 0 and let computer to optimize it 
        #features[:, :3, 0 ] = fused_color  
        #features[:, 3:, 1:] = 0.0

        #dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(points)).float().cuda()), 0.0000001)
        #scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        #rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        #rots[:, 0] = 1
        #opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # if self.debug:
        #     # easy for visualization
        #     colors = np.zeros_like(colors)
        #     opacities = inverse_sigmoid(0.9 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))


 
        #initialize all params 
        #fused_point_cloud[:,0]=torch.rand(fused_point_cloud.shape[0])-0.5
        #fused_point_cloud[:,1]=torch.rand(fused_point_cloud.shape[0])-0.5
        
        ########for debug##############
        #fused_point_cloud[:,0]=0.0919
        #fused_point_cloud[:,1]=0.4923
        #fused_point_cloud[:,2]=0.4180
        ###############################

        

        fused_point_cloud[:,0]=torch.rand(fused_point_cloud.shape[0])-0.5
        fused_point_cloud[:,1]=torch.rand(fused_point_cloud.shape[0])-0.5
        fused_point_cloud[:,2]=torch.rand(fused_point_cloud.shape[0])-0.5

        
        scales = torch.log (torch.ones((fused_point_cloud.shape[0]), device="cuda")*0.01)

        #scales = torch.ones((fused_point_cloud.shape[0]), device="cuda")*0.01

        opacity_sigma = torch.ones((fused_point_cloud.shape[0]), device="cuda")*1.0 # after sigmoid which is around 0.1

        funny_point_params =  torch.ones((fused_point_cloud.shape[0],4), device="cuda")*0.5
        

        #####for debug
        # camera_space_p0= torch.tensor([0.0 , 0.2 , 1 , 1.0],device="cuda")
        # fused_point_cloud[0]= (camera_space_p0 @ (camera.c2w.permute(1,0)))[:3]
        # fused_color[0]= torch.tensor([0.0 , 100.0 , 0.0],device="cuda")  
        # opacity_sigma[0] = torch.tensor(1.0,device="cuda") 
        # scales[0] = torch.tensor(0.1,device="cuda") 

        #color 

        #fused_color[:,:]=0.0
        #fused_color[:,:]= 0.0 # after sigmoid it is around 0.0

        #rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")

        #rots[:, 0] = 1
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
    
        #self._rotation = nn.Parameter(rots.requires_grad_(True))


        #self._color =  nn.Parameter(fused_color.requires_grad_(True))
         
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
         
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._opacity_sigma = nn.Parameter(opacity_sigma.requires_grad_(True))
        
        self.funny_point_params = nn.Parameter(funny_point_params.requires_grad_(True))
   
        return self
    
    @property
    def get_opacity_sigma(self):
        #return torch.relu(self._opacity_sigma)
        return torch.sigmoid(self._opacity_sigma)


    @property
    def get_funny_point_params(self):
        return self.funny_point_params
    @property
    def get_scaling(self):
        #return torch.relu(self._scaling)
        #return torch.relu(torch.clip(self._scaling, max=0.05))  ## debug
        return  torch.exp(self._scaling)
        #return  self._scaling

    @property
    def get_xyz(self):
        return  self._xyz

    # @property
    # def get_color(self):
    #     #return torch.relu(self._color)
    #     return torch.sigmoid(self._color)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
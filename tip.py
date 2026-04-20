import torch

critial_points = torch.tensor([2,3])

x = torch.tensor([
[
[1, 2],[3, 4]
],
[
[8, 9],[10, 11]
],
[
[14,15],[16,17]
]
])
index = x>=3
indices = torch.argwhere(x >= 3)
print(indices)
#################
last_elements = indices[:, -1]
print(last_elements)

x[index]=x[index]+critial_points[last_elements]
print(x)

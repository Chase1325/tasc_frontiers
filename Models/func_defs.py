from numpy import dtype
import torch
import time
import plotly.graph_objects as go
from torch._C import device
import numpy as np

#@torch.jit.script
def constant(x, device='cpu'):
    return torch.ones(x.size()[0], dtype=torch.float64, device=torch.device(device)).unsqueeze(0).t()

#@torch.jit.script
def eggholder(v, device=None):
    (x, y) = torch.tensor_split(v, 2, dim=1)
    return -(y + 47)*torch.sin(torch.sqrt(torch.abs(y+0.5*y+47))) - x*torch.sin(torch.sqrt(torch.abs(x-(y+47)))) + 960

def easom(v, device=None):
    (x, y) = torch.tensor_split(v, 2, dim=1)
    return -torch.cos(x)*torch.cos(y)*torch.exp(-torch.pow(x-3.14159, 2)-torch.pow(y-3.14159, 2))

def bird(v, device=None):
    (x, y) = torch.tensor_split(v, 2, dim=1)
    return torch.sin(x)*torch.exp(torch.pow(1-torch.cos(y), 2)) + torch.cos(y)*torch.exp(torch.pow(1-torch.sin(x),2)) + torch.pow(x-y, 2) + 107

def holder_table(v, device=None):
    (x, y) = torch.tensor_split(v, 2, dim=1)
    f1 = torch.sin(x)*torch.cos(y)
    f2 = torch.exp(torch.abs(1 - torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))/3.14159))
    return -torch.abs(f1*f2)
    #return -torch.abs(torch.sin(x)*torch.cos(y)*torch.exp(torch.abs(1-(torch.sqrt(x**2 + y**2)/3.14159))))

def rastragin(v, device=None):
    (x, y) = torch.tensor_split(v, 2, dim=1)
    return x**2 - 10*torch.cos(2*3.14159*x) + y**2 - 10*torch.cos(2*np.pi*y) + 20

def drop_wave(v, device=None):
    (x, y) = torch.tensor_split(v, 2, dim=1)
    return (1 - (1 + torch.cos(12*torch.sqrt(x**2 + y**2))) / (0.5*(x**2 + y**2) + 2))

def bukin6(v, device=None):
    (x, y) = torch.tensor_split(v, 2, dim=1)
    return 100*torch.sqrt(torch.abs(y-0.01*x**2))+0.01*torch.abs(x+10)

if __name__ == "__main__":
    
    
    i = torch.linspace(-512, 512, 100, device=torch.device('cuda'))
    j = torch.linspace(-512, 512, 100, device=torch.device('cuda'))


    start = time.time()
    gx, gy = torch.meshgrid(i, j)

    x = torch.column_stack((gx.ravel(), gy.ravel()))

    x2 = x[2230:26245, :]
    #print(x)
    
    f = constant(x)
    #print(f, f.size())
    f = bird(x)
    f = easom(x)
    
    f = eggholder(x)
    
    
    f2 = eggholder(x2)
    
    #print(eggholder(x))

    print(time.time()-start)
    #print(f, x)
    field = f.cpu()
    path = f2.cpu()
    gx_Cpu = i.cpu()
    gy_Cpu = j.cpu()
    (xi, yi) = torch.tensor_split(x, 2, dim=1)
    xc = xi.cpu()
    yc = yi.cpu()
    #fig = go.Figure(data=go.Contour(z=field.reshape(100,100).t(), dx=1024/100, x0=-512, dy=1024/100, y0=-512, colorscale='viridis'))
    fig = go.Figure(data=[go.Surface(z=field.reshape(100,100).t(), x=gx_Cpu.ravel(), y=gy_Cpu.ravel(), colorscale='Viridis', opacity=0.75)])
    fig.update_layout(scene = dict(
        zaxis = dict(nticks=2, range=(min(field).item(), 5*max(field).item()))
    ), scene_aspectratio=dict(x=1, y=1, z=0.5))
    fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor='white', project_z=True))
    print(gx_Cpu.ravel()[30:45], gy_Cpu.ravel()[30:45], path.squeeze)
    line = go.Scatter3d(x=xc.ravel()[2230:2645], y=yc.ravel()[2230:2645], z=path.squeeze(), line=dict(color='white', width=5), mode='lines')
    fig.add_trace(line)
    fig.show()
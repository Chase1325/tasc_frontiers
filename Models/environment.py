from scipy.sparse import csr
import torch
import time
import numpy as np
import scipy.sparse as s
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from torch._C import device, dtype
import plotly.graph_objects as go


class Environment(object):
    def __init__(self, f, domain, res, start, goal, connection=8, device='cpu'):
        self.device = device

        #Field specific
        self.domain = domain
        self.res = res
        self.dp = (self.domain[1]-self.domain[0])/(self.res-1)
        self.create_wsp()
        self.create_field(f)
        self.s = self.get_closest_idx(start)
        self.g = self.get_closest_idx(goal, True)

        #Create initial paths and vars
        self.idvec = torch.zeros(self.res**2, dtype=torch.int8, device=torch.device(self.device))

        #Graph and Plan
        self.create_adjMat(connection) 
        self.path_plan()

    def get_closest_idx(self, x, goal=False):
        if goal:
            self.dvec = torch.cdist(self.wsp, torch.tensor([x], dtype=torch.float64, device=torch.device(self.device)))
            return torch.argmin(self.dvec)
        else:
            return torch.argmin(torch.cdist(self.wsp, torch.tensor([x], dtype=torch.float64, device=torch.device(self.device))))
    def create_wsp(self):
        x = torch.linspace(self.domain[0], self.domain[1], self.res, dtype=torch.float64, device=torch.device(self.device))
        gx, gy = torch.meshgrid(x, x)
        self.wsp = torch.column_stack((gx.ravel(), gy.ravel()))

    def create_field(self, f):
        self.field = f(self.wsp, device=self.device)

    def create_adjMat(self, connection=8):

        if connection==4:
            d1 = np.tile(np.append(np.ones(self.res-1, dtype=np.float64), [0]), self.res)[:-1]
            d2 = np.ones(self.res*(self.res-1), dtype=np.float64)
            upper_diags = s.diags([d1, d2], [1, self.res])
            adjMat = (upper_diags + upper_diags.T).toarray()
            self.adjMat = torch.from_numpy(adjMat).to(self.device)*self.dp

        if connection==8:
            d1 = np.tile(np.append(np.ones(self.res-1), [0]), self.res)[:-1]
            d2 = np.append([0], np.sqrt(2)*d1[:self.res*(self.res-1)])
            d3 = np.ones(self.res*(self.res-1))
            d4 = d2[1:-1]
            upper_diags = s.diags([d1, d2, d3, d4], [1, self.res-1, self.res, self.res+1])
            adjMat = (upper_diags + upper_diags.T).toarray()
            self.adjMat = torch.from_numpy(adjMat).to(self.device)*self.dp

    def path_plan(self, est_path=False):
        

        if not est_path:
            weightVec = 1e-16*self.dvec + self.field
            fieldweight = self.adjMat  * (weightVec + weightVec.t())*0.5

            if self.device == 'cuda':
                f_cpu = fieldweight.cpu().numpy()
                g = self.g.cpu().numpy()
                path_cost, predecesors = shortest_path(csr_matrix(f_cpu), method="D", directed=False, indices=g, return_predecessors=True)
            else:
                path_cost, predecesors = shortest_path(csr_matrix(fieldweight), method="D", directed=False, indices=self.g, return_predecessors=True)
            self.incidenceVec = torch.zeros(self.res**2)

            if est_path:
                self.est_cost = path_cost[self.s]*self.dp
            else:
                self.path_cost = path_cost[self.s]*self.dp
            
            #Path Incidence (weighted for diagonals)
            self.p_true = torch.zeros_like(self.field, device=torch.device(self.device))
            indx = self.s.item()
            v = [indx]
            #jndx = predecesors[indx]
            while indx != self.g:
                jndx = predecesors[indx].item()
                delta = self.adjMat[indx][jndx]
                self.p_true[[indx, jndx]] += delta*0.5
                v.append(jndx)
                indx = jndx
        
            self.v_true = torch.tensor(v, device=torch.device(self.device))
    

        
        #print(torch.mm(self.field.t(), (self.p_true)))
        '''
        # Trial Get path and plot on surface
        zg = self.field[self.v_true, :]
        zc = zg.to('cpu')
        xg, yg  = torch.tensor_split(self.wsp[self.v_true, :], 2, dim=1)
        xc = xg.to('cpu')
        yc = yg.to('cpu')
        field = self.field.cpu()

        xg_, yg_  = torch.tensor_split(self.wsp, 2, dim=1)
        cx, cy = np.linspace(self.domain[0], self.domain[1], self.res), np.linspace(self.domain[0], self.domain[1], self.res)
        #print(field.reshape(100,100).t(), cx.ravel(), cy.ravel())
        fig = go.Figure(data=[go.Surface(z=field.reshape(self.res,self.res).t(), x=cx.ravel(), y=cy.ravel(), colorscale='Viridis', opacity=1)])
        fig.update_layout(scene = dict(
            zaxis = dict(nticks=2, range=(min(field).item()-1, 5*max(field).item()+1))), scene_aspectratio=dict(x=1, y=1, z=0.65))
        #print(xc, yc, zc)
        line = go.Scatter3d(x=xc.ravel(), y=yc.ravel(), z=zc.ravel()+1e-3, line=dict(color='white', width=5), mode='lines')
        #lower = go.Surface(z=field.reshape(self.res,self.res).t()-1, x=cx.ravel(), y=cy.ravel(), colorscale='Blues', opacity=0.25)
        #upper = go.Surface(z=field.reshape(self.res,self.res).t()+1, x=cx.ravel(), y=cy.ravel(), colorscale='Reds', opacity=0.25)
        fig.add_trace(line)
        #fig.add_trace(upper)
        fig.show()

        fig = go.Figure(data=[go.Heatmap(z=field.reshape(self.res,self.res).t(), x=cx.ravel(), y=cy.ravel(), colorscale='Viridis', zsmooth='best')])
        line = go.Scatter(x=xc.ravel(), y=yc.ravel(), line=dict(color='white', width=5), mode='lines')
        fig.add_trace(line)
        fig.show()
        '''
        

    def print_pretty(self):
        print(" Domain: {}\n Resolution: {}\n Start: {} @ {}\n Goal: {} @ {}\n".format(self.domain, 
                                                                                       self.res, 
                                                                                       self.s, self.wsp[self.s],
                                                                                       self.g, self.wsp[self.g]))
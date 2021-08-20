import numpy as np
import random
import time
import argparse
from numpy.core.numeric import indices
import torch
from scipy.sparse.csgraph import shortest_path, dijkstra
from scipy.sparse import csr_matrix
from torch._C import StringType, device

#from Methods.Utility import createGoals, getPathVar, encapsulateData, getStats, maskField
import Models.func_defs as fds
from Models.environment import Environment
#from Methods.Uniform_Cost_Search import opt_path
#from Methods.GP_REGRESSION import GP_REGRESSION

#from Models.Sensor_Network import SensorNetwork
#from Models.Graph import createCompact2DGraph
#from Models.Threat import GaussThreatField
#from Models.Environment import Environment

#import Visuals.Visualize as viz

np.set_printoptions(suppress=True)

def TASC(E, S, G, i_s, i_g, eps, visualize=False, method='direct', doCluster=False):

    startTime = time.time()

    #Initialize Variables
    f = np.zeros(E.Ng)                              # Estimated threat field
    P = np.eye(E.Ng)*10000
    #P = np.diag(np.random.rand(E.Ng))*10000         # Covariance Matrix
    I = np.array([], dtype=np.int16)                # Identified vertices
    X_k = o_k = n_k = B_ = B_n = B_o = None
    iteration = 0
    observations = 0
    identified = False

    #Find true best path
    pi_t, path_t, Ex_t, _ = opt_path(E, E.true_field)

    #Find initial path
    s_time = time.time()
    pi_s, path_s, Ex_s, _ = opt_path(E, E.workspace, heuristic='distance')
    print(time.time()-s_time)

    #Find initial path variance
    VAR_path = getPathVar(pi_s, P, E.dp)

    while (VAR_path > eps) or not identified:
        #Find Sensor Configurations and make observations
        s_time = time.time()
        iterations = S.SENSOR_CONFIG(E, f, P, pi_s, path_s, E.dp, term=eps, id=I, method=method, alts=E.alts, doCluster=doCluster)
        print('Sensor Config', time.time()-s_time)
        x, y, noise, obs, I_k, Beta, Beta_n, Beta_o = S.getMeasurements(E.threat_field, E.workspace)

        #Encapsulate Data Method
        s_time = time.time()
        X_k, o_k, n_k, B_, B_o, B_n, I = encapsulateData(x, y, noise, obs, I_k, Beta, Beta_n, Beta_o,
                                                         X_k=X_k, o_k=o_k, n_k=n_k, B_=B_, B_n=B_n, B_o=B_o, I=I)
        

        #Find Field and Covariance Matrix with GPR
        train_pt = np.concatenate((X_k, B_), axis=0)
        train_label = np.concatenate((o_k, B_o))
        train_noise = np.concatenate((n_k, B_n))
        print('Concat time', time.time()-s_time)

        f, P = GP_REGRESSION(train_pt, train_label, train_noise, E)
        
        #ID incidence masking and correct negative values (UTILITY)
        f_, f_adj = maskField(I, f, E.Ng, id=True)

        #Find est optimal path, path variance, true est path cost, est path cost, and # of observations made
        s_time = time.time()
        pi_s, path_s, _, _ = opt_path(E, f_adj)
        print('Path Plan:', time.time()-s_time)
        Ex_s = E.dp*np.sum(f[pi_s[1:]])
        VAR_path = getPathVar(pi_s, P, E.dp)

        #Report on estimation error, incurred error and % identified field, and if path identified or not
        Ex_s_true = E.dp*sum([E.threat_field.threat_value(pt[0], pt[1]) for pt in path_s[1:]])
        iteration += 1
        observations += S.active_Ns
        if np.all(np.in1d(pi_s[1:], I)):
            identified = True
        else:
            identified = False

        E_err, Inc_err, I_per = getStats(Ex_s, Ex_s_true, Ex_t, I, E.Ng)
        print(VAR_path, E_err, Inc_err, I_per)

        # OPTIONAL: Visualize field updates
        if visualize:

            #print(S.network)
            #viz.draw_fov(E, I_k, S.network.reshape(-1, 3))
            #viz.draw_fig_3d(E, f.reshape(E.Ng_axis, E.Ng_axis))
            viz.draw_env_iteration_paper2(E, path_t, path_s, f_.reshape(E.Ng_axis, E.Ng_axis),
                                                            P.diagonal().reshape(E.Ng_axis, E.Ng_axis),
                                                            sense=S.network.reshape((S.active_Ns, 3)), id=I)


    #Return the data
    runtime = time.time() - startTime
    return iteration, observations, E_err, Inc_err, I_per, runtime

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Initialization settings for TASC')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--Ns', type=int, default=5, help='number of sensors')
    parser.add_argument('--start', type=float, nargs="+", default=(-1,-1))
    parser.add_argument('--goal', type=float, nargs="+", default=(1,1))
    parser.add_argument('--res', type=int, default=100, help='number of vertices per axis (rounds to nearest square root per axis)')
    parser.add_argument('--eps', type=float, default=.01, help='Termination threshold value')
    parser.add_argument('--domain', type=float, default=(-1, 1), nargs="+", help='physical axis size of workspace (-1, 1)')
    parser.add_argument('--viz', type=bool, default=True, help='Visualize outputs or not')

    # Sampling Parameter
    parser.add_argument('--alts', type=int, default=10, help='Number of alternate paths for breadth strategy')

    # Random field parameters
    parser.add_argument('--Np', type=int, default=50, help='number of threat parameters')
    parser.add_argument('--intensity', type=float, default=25.0, help='threat field intensity (must be strictly positive)')
    #parser.add_argument('--offset', type=float, default=1.0, help='threat field offset (must be strictly positive)')

    parser.add_argument('--device', type=str, default='cpu')
    
    
    #parser.add_argument('--method', type=str, default='direct', help='CSCP strategy to use [direct, depth, breadth]')
    #parser.add_argument('--cluster', type=int, default=0, help='Perform clustering instead of direct optimization for sensor configuration')
    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    start = time.time()
    E = Environment(fds.eggholder, args.domain, args.res, args.start, args.goal, device=args.device)
    E.path_plan()
    E.path_plan()
    print(time.time()-start)

    start = time.time()
    for i in range(10):
        E.path_plan()
    print(time.time()-start)

    #i_s, i_g = createGoals((args.start), (args.goal), res=args.res, size=args.size, seed=args.seed)
    '''
    c = GaussThreatField(Np=args.Np, offset=args.offset, intensity=args.intensity, size=args.size, seed=args.seed)
    E = Environment((args.size, args.size), args.res, c, offset=args.offset, alts=args.alts, start=i_s, goal=i_g)
    G = createCompact2DGraph(E.workspace)
    S = SensorNetwork(args.Ns)
    E.graph = G
    #print(i_s, i_g)
    #print(args.cluster)

    #viz.draw_env_3d(E)
    #Find true best path
    #pi_t, path_t, Ex_t, _ = opt_path(E, E.true_field)
    #viz.draw_env_2d(E, draw_workspace=True, true_path=path_t)
    iters, obs, E_err, Inc_err, I_per, runtime = TASC(E, S, G, i_s, i_g, args.eps, args.viz, method=args.method, doCluster=args.cluster)
    print('{} Iters; {} Placements; {} Est-err; {} Inc-err; {} Id; {} secs;'.format(iters, obs, E_err, Inc_err, I_per, runtime))
    '''

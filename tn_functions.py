import tensornetwork as tn
import numpy as np
import matplotlib.pyplot as plt


def haar_unitary(N):
    A, B = np.random.normal(size = (N,N)), np.random.normal(size = (N,N))
    Z = A + 1j*B
    
    Q, R = np.linalg.qr(Z)
    Lambda = np.diag([R[i,i]/np.abs(R[i,i]) for i in range(N)])
    
    U = np.matmul(Q, Lambda)
    U_dag = np.transpose(np.conjugate(U))
    
    return U, U_dag


def check_unitary(U_, U_dag_):
    UU_dag = np.matmul(U_, U_dag_)
    U_dagU = np.matmul(U_dag_, U_)
    
    N_U = np.shape(U_)[0]
    
    if np.any(np.abs(UU_dag - np.eye(N_U)) > 1e-12):
        print('Error')
    
    if np.any(np.abs(U_dagU - np.eye(N_U)) > 1e-12):
        print('Error')


def normalization(obj):
    obj_norm = obj/np.sqrt(np.sum(np.abs(obj**2)))
    return obj_norm



def check_normalization(obj):
    if not np.allclose(np.sqrt(np.sum(np.abs(obj**2))), 1, atol = 1e-12):
        print('Error not normalized')
        


def block(*dimensions):
    '''Construct a new matrix for the MPS with random numbers from 0 to 1'''
    size = tuple([x for x in dimensions])
    data = np.random.normal(size=size) + 1j*np.random.normal(size=size)
    data = normalization(data)
    return data



def create_MPS(rank, dim, bond_dim, bond_dim_ent, plot = False):
    '''Build the MPS tensor'''        
    mps = [tn.Node( block(dim[0], bond_dim_ent[0], bond_dim_ent[1]), name = 'T1', axis_names = ['d1', 'b12', 'b13'])] + [tn.Node( block(bond_dim_ent[0], dim[1], bond_dim), name = 'T2', axis_names = ['b21', 'd2', 'b23'])] + [tn.Node( block(bond_dim_ent[1], bond_dim, dim[2]), name = 'T3', axis_names = ['b31', 'b32', 'd3'])]

    mps[0][1]^mps[1][0]
    mps[0][2]^mps[2][0]
    mps[1][2]^mps[2][1]

    if len(mps) != rank:
        print('Rank wrong')
    for i in range(rank):
        if (mps[i].get_dimension(i) != dim[i]):
            print(f"Physical dimension of tensor {i} wrong")

    if plot:
        display(tn.to_graphviz(mps))
        
    return mps



def contract_T(mps, rank, dim, bond_dim, bond_dim_ent, plot = False):        
    T_1 = tn.contract_between(mps[1],mps[2])
    #display(tn.to_graphviz([T_1]))
    
    T = tn.contract_between(mps[0], T_1)
    
    T.set_name('T')
    T.add_axis_names(['d1','d2','d3'])
    if plot:
        display(tn.to_graphviz([T]))

    if len(T.shape) != rank:
        print('T rank wrong')
    if (T.shape != tuple(dim)):
        print('T physical dimension wrong')
        print(T.shape)
        print(phys_dim)

    data_update = normalization(T.tensor)
    T.set_tensor(data_update)

    check_normalization(T.tensor)
        
    return T



def check_tensor_norm(T, plot = False):
    T_ = tn.replicate_nodes([T])[0]   #copy 1 of the tensor
    T__ = tn.replicate_nodes([T])[0]  #copy 2 of the tensor
    T__.tensor = np.conjugate(T__.tensor)   #make copy 2 the complex conjugate of copy 1
    T__.set_name(T_.name + str('*'))

    for i in range(len(T_.get_all_edges())):
        edge_ = T_[i] ^ T__[i]
    
    if plot:
        display(tn.to_graphviz([T__,T_]))
    
    res = tn.contract_between(T_,T__)
    res.set_name('Norm')
    if plot:
        display(tn.to_graphviz([res]))

        print('Norm of the tensor: ', res.tensor)

    if not np.allclose(res.tensor, 1):
        print('Error, tensor not normalized')
        print('Norm is: ', res.tensor)

    return res



def check_unitary_tensor(U, plot = False):
    U_d = tn.replicate_nodes([U])[0]
    U_d.tensor = np.conjugate(U.tensor)
    if plot:
        display(tn.to_graphviz([U,U_d]))
    
    U[0] ^ U_d[0]
    if plot:
        display(tn.to_graphviz([U,U_d]))

    res = tn.contract_between(U, U_d)
    res_tensor = res.tensor

    if not np.allclose(res_tensor, np.eye(U.get_dimension(0)), atol = 1e-12):
        print('Error, matrix not unitary')



def create_U(T, fix_seed = True, plot = False):   
    if fix_seed:
        np.random.seed(123456789)
        
    N = T.get_dimension('d1')
    U_matrix, U_dag_matrix = haar_unitary(N)
    check_unitary(U_matrix, U_dag_matrix)
    
    U = tn.Node(U_matrix, name='U', axis_names=['d1','d1_'])
    check_unitary_tensor(U)
    if plot:
        display(tn.to_graphviz([U]))
    
    return U



def contraction_TU(T, U, split_d4, split_d5, plot = False):
    T_copy = tn.replicate_nodes([T]) [0]
    dim = T.shape

    #Split U
    leg1, leg2 = tn.split_edge(U['d1_'], [split_d4, split_d5])
    U.reorder_edges([U.get_edge(0), leg1, leg2])
    U.add_axis_names(['d1','d4','d5'])
    U['d4'].set_name('d4')
    U['d5'].set_name('d5')
    
    if ((U.get_dimension('d4') != split_d4) & (U.get_dimension('d5') != split_d5)):
        print('Error in splitting over d4, d5')

    if plot:
        display(tn.to_graphviz([U]))

    #Contract T and U
    edge_d1 = T['d1'] ^ U['d1']
    if plot:
        display(tn.to_graphviz([T,U]))
    
    TU = tn.contract(edge_d1)
        
    if ([TU[i].name for i in range(len(TU.get_all_edges()))] != ['d2','d3', 'd4', 'd5']):
        print('Error in assigning names')
        print('Real names: ', [TU[i].name for i in range(len(TU.get_all_edges()))])
    
    TU.add_axis_names(['d2','d3', 'd4', 'd5'])
    TU.set_name('TU')
    
    if ((TU.get_dimension('d4') != split_d4) & (TU.get_dimension('d5') != split_d5)):
        print('Error in contraction over d4, d5')
    if ((TU.get_dimension('d2') != dim[1]) & (TU.get_dimension('d3') != dim[2])):
        print('Error in contraction over d2, d3')

    if plot:
        display(tn.to_graphviz([TU]))
        
    return TU, T_copy



def apply_SVD(TU, max_truncation_error, plot = False):
    TU_copy = tn.replicate_nodes([TU])[0]
    
    U_svd, sing_val, V_svd_d, _ = tn.split_node_full_svd(TU, left_edges = [TU['d4'], TU['d2']], right_edges = [TU['d5'], TU['d3']], max_truncation_err = 0)  #d4d2d5d3-2031

    if plot:
        display(tn.to_graphviz([U_svd, sing_val, V_svd_d]))

    U_svd_copy, sing_val_copy, V_svd_d_copy = tn.replicate_nodes([U_svd,sing_val,V_svd_d])
    TU_reconstruted_partial = tn.contract_between(sing_val_copy, V_svd_d_copy)
    TU_reconstructed = tn.contract_between(U_svd_copy, TU_reconstruted_partial)

    TU_reconstructed.reorder_axes([1,3,0,2])       #check shape original matrix and reconstructed, order the dimensions of the reconstructed to match original
    if not np.allclose(TU_reconstructed.tensor,TU_copy.tensor, atol = 1e-12):
        print('Error in SVD decomposition')
        
    return U_svd, sing_val, V_svd_d, TU_copy



def apply_SVD_horizontal(TU, max_truncation_error, plot = False):
    TU_copy = tn.replicate_nodes([TU])[0]
    
    U_svd, sing_val, V_svd_d, _ = tn.split_node_full_svd(TU, left_edges = [TU['d2'], TU['d3']], right_edges = [TU['d4'], TU['d5']], max_truncation_err = 0)

    if plot:
        display(tn.to_graphviz([U_svd, sing_val, V_svd_d]))
    
    U_svd_copy, sing_val_copy, V_svd_d_copy = tn.replicate_nodes([U_svd,sing_val,V_svd_d])
    TU_reconstruted_partial = tn.contract_between(sing_val_copy, V_svd_d_copy)
    TU_reconstructed = tn.contract_between(U_svd_copy, TU_reconstruted_partial)

    if not np.allclose(TU_reconstructed.tensor,TU_copy.tensor, atol = 1e-12):
        print('Error in SVD decomposition')
        
    return U_svd, sing_val, V_svd_d, TU_reconstructed



def entropy(sing_val, print_result = False):
    s_val = np.diag(sing_val.tensor).real
    S = -np.sum(s_val**2 * np.log(s_val**2))

    if print_result:
        if S < 1e-12:
            S = 0
        print('Entropy S = ', S)
    return s_val, S



def find_split(x):
    L=[]
    mcm = np.arange(1,x+1)
    for i in mcm:
        for j in mcm:
            if i*j==x:
                L.append((i,j))
                
    return L



def which_tensor(ent):
    if ent[0][1]<0.01 and ent[0][0]>0.01 and ent[-1][0] >0.01:
        string='d2-d3'
    elif ent[0][0]<0.01 and ent[0][1]>0.01 and ent[-1][0]>0.01:
        string='d1-d3'
    elif ent[0][0]>0.01 and ent[0][1]>0.01 and ent[-1][0]<0.01:
        string='d1-d2'
    elif ent[0][0]<0.01 and ent[0][1]<0.01 and ent[-1][0]<0.01:
        string='no entanglement'
    else:
        string='more than one couple'
    
    return string
    



def find_entangled_systems(T, print_result = False):
    check_tensor_norm(T)
    
    d1 = T.shape[0]
    L = find_split(d1)
    print('Possible splitting combination:, \n', L)
    
    entropy_values = []
    rank_list = []
    rank_list_ = []
    for i in range(len(L)):
        if i > 0:
            T = tn.replicate_nodes([T_copy])[0]
            T_copy = 0
        
        U = create_U(T)
        split_d4 = L[i][0]
        split_d5 = L[i][1]

        TU, T_copy = contraction_TU(T, U, split_d4, split_d5)
        
        U_svd, sing_val, V_svd_d, TU_copy = apply_SVD(TU, max_truncation_error = 0)
        s_values1, S1 = entropy(sing_val)
        
        if S1<1e-2:
            S1=0
        if print_result:
            print("\nEntropy values for vertical SVD for d4=", split_d4, "and d5=", split_d5, "is S=", S1)
            
        U_svd_, sing_val_, V_svd_d_, __ = apply_SVD_horizontal(TU_copy, max_truncation_error = 0)
        s_values2, S2 = entropy(sing_val_)
        
        if S2<1e-2:
            S2=0
        if print_result:
            print("\nEntropy values for horizontal SVD for d4=", split_d4, "and d5=", split_d5, "is S=", S2)
            
        entropy_values.append((S1,S2))
        
        rank = np.diag(sing_val.tensor.real)
        rank_list.append(len(rank[rank>1e-14]))
    
        rank_ = np.diag(sing_val_.tensor.real)
        rank_list_.append(len(rank_[rank_>1e-14]))

        T, U, TU, U_svd, sing_val, V_svd_d, TU_copy, U_svd_, sing_val_, V_svd_d_, __ = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        
    print(which_tensor(entropy_values))

    return entropy_values



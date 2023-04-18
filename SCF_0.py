import numpy as np
from scipy.linalg import fractional_matrix_power


# 将特征值与特征矢量按特征值从小到大重新排序
def sort_eigs(eigvecs, eigvals):
    """
    将特征值与特征矢量按特征值从小到大重新排序

    input:
    eigvecs:本征矢量
    eigval:本征值

    output:
    从小到大重新排序特征值与特征矢量
    """
    idx = eigvals.argsort()
    sorted_eigvals = eigvals[idx]
    sorted_eigvecs = eigvecs[:, idx]
    return sorted_eigvecs, sorted_eigvals


# 形成核心Hamilton矩阵
def Hcore(kinetic_matrix, nuclear_matrix, ele_num):
    """
    形成核心Hamilton矩阵

    input:
    kinetic_matrix:动能矩阵
    nuclear_matrix:核势能矩阵的字典

    output:
    核心Hamilton矩阵
    """
    H = kinetic_matrix
    for i in range(1, ele_num + 1):
        H += nuclear_matrix[i]
    return H


# 总能量
def fock_energy(Density, Hcore, Fock):
    """
    总能量

    input:
    Density:密度矩阵
    Hcore:核心Hamilton矩阵
    Fock:Fock矩阵

    output：
    体系总能量
    """
    E = np.trace(np.dot(Hcore + Fock, Density))
    return E


# 构造G矩阵
def Build_Coulomb_Exchange(Density, eri_mat):
    """
    构造G矩阵

    input:
    Density:密度矩阵
    eri_mat:电子排斥积分矩阵

    output:
    G矩阵
    """
    dim = Density.shape[0]
    G = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    # Swaping basis set 2 and 3 to get exchange integral
                    G[i, j] = G[i, j] + Density[k, l] * (
                            2 * eri_mat[i * dim + j, k * dim + l] - eri_mat[i * dim + l, k * dim + j])

    return G


# 构造电子密度矩阵
def Build_Density(wave_coeff, N):
    """
    构造电子密度矩阵

    input:
    wave_coeff:求得的特征矢量矩阵
    N:总电子数

    output：
    电子密度矩阵
    """
    dim = wave_coeff.shape[0]
    D = np.zeros((dim, dim))
    for n in range(dim):
        for m in range(dim):
            for i in range(int(N / 2)):
                D[n, m] = D[n, m] + wave_coeff[n, i] * wave_coeff[m, i]

    return D


# 核与核之间的互斥能
def rep_nucl_nucl(info, nuclear_charge):
    """
    核与核之间的互斥能

    input:
    info: a dict about all kinds of information
    nuclear_charge:一个包含各个元素核电荷的字典

    output:
    核与核之间的互斥能
    """
    E = 0
    for i in range(1, len(info.keys())):
        for j in range(i+1, len(info.keys())):
            coord_i = np.array(list(map(float, info[i]['coordinate'])))
            coord_j = np.array(list(map(float, info[j]['coordinate'])))
            distance = np.linalg.norm(coord_i - coord_j)
            charge_i = nuclear_charge[info[i]['element']]
            charge_j = nuclear_charge[info[j]['element']]
            E += float(charge_i) * float(charge_j)/distance
    return E


# 自洽场过程，最终输出一个包含density_matrix,orb_coeff,orb_energy,total_energy,Fock_matrix,Hcore_matrix数据的字典
def SCF(Hcore_mat, eri_mat, S, number_electrons, info, nuclear_charge):
    """
    自洽场过程，最终输出一个包含density_matrix,orb_coeff,orb_energy,total_energy,Fock_matrix,Hcore_matrix数据的字典

    input:
    Hcore:核心Hamilton矩阵
    eri_mat:电子排斥积分矩阵
    S:重叠矩阵
    number_electrons:电子数
    info:a dict about all kinds of information
    nuclear_charge:一个包含各个元素核电荷的字典

    output:
    输出一个包含density_matrix,orb_coeff,orb_energy,total_energy,Fock_matrix,Hcore_matrix数据的字典
    """
    maxcycles = 200
    converged = 0
    ncycle = 0
    E = np.zeros(maxcycles)

    eigvals, V = np.linalg.eigh(S)
    D = np.diag(eigvals)

    # X = V @ fractional_matrix_power(D, -0.5) @ V.T
    X = V @ fractional_matrix_power(D, -0.5)

    Fock_mat = Hcore_mat
    trans_fock = X.T @ Fock_mat @ X

    orb_energy, trans_coeff = np.linalg.eig(trans_fock)
    wave_coeff = X @ trans_coeff

    wave_coeff, orb_energy = sort_eigs(wave_coeff, orb_energy)
    Density = Build_Density(wave_coeff, number_electrons)

    while ncycle < maxcycles - 1 and converged != 1:
        ncycle += 1
        print('cyc', ncycle)
        G = Build_Coulomb_Exchange(Density, eri_mat)
        Fock_mat = Hcore_mat + G
        trans_fock = X.T @ Fock_mat @ X

        orb_energy, trans_coeff = np.linalg.eig(trans_fock)
        wave_coeff = X @ trans_coeff

        wave_coeff, orb_energy = sort_eigs(wave_coeff, orb_energy)
        Density = Build_Density(wave_coeff, number_electrons)

        E[ncycle] = fock_energy(Density, Hcore_mat, Fock_mat)

        if (ncycle > 1) and (abs(E[ncycle] - E[ncycle - 1]) < 10e-11):
            converged = 1
            print('SCF has converged!')
    E[ncycle] += rep_nucl_nucl(info, nuclear_charge)
    scf_info = {'density_matrix': Density, 'orb_coeff': wave_coeff, 'orb_energy': orb_energy,
                'total_energy': E[ncycle], 'Fock_matrix': Fock_mat, 'Hcore_matrix': Hcore_mat}
    return scf_info

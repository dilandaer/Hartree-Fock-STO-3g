import numpy as np
import scipy.special as spec
import mpmath as mp


# General Cartesian Gaussian normalization factor
def norm(ax, ay, az, alpha):
    """
    General Cartesian Gaussian normalization factor

    input:
        ax: Angular momentum x
        ay: Angular momentum y
        az: Angular momentum z
        alpha: Gaussian exponential value

    output:
        Normalization coefficient N
    """
    # Calculate normalization coefficient
    N = (2 * alpha / np.pi) ** (3. / 4.)
    N *= (4 * alpha) ** ((ax + ay + az) / 2)
    N /= np.sqrt(spec.factorial2(2 * ax - 1) * spec.factorial2(2 * ay - 1) * spec.factorial2(2 * az - 1))
    return N


# gaussian product, 将两个高斯函数转化为一个高斯函数
def gaussian_product(alpha, beta, Ra, Rb):
    """
    input:
    alpha, beta are the exponential coefficients of Gaussian 1 and 2
    Ra and Rb are the centers of Gaussian 1 and 2
    output:
    p: Exponential coefficient of Gaussian product
    Rp: the center of Gaussian product
    K: Gaussian product coefficient

    source: SAZOB, Guangxian Xu(Quantum Chemistry)
    """

    # Transform centers in number array
    Ra = np.asarray(Ra)
    Rb = np.asarray(Rb)

    # Exponetial coefficient of Gaussian product
    p = alpha + beta

    # Calculate the center of Gaussian product
    Rp = (alpha * Ra + beta * Rb) / p

    # Calculate the coefficient of Gaussian product
    K = np.dot(Ra - Rb, Ra - Rb)
    K *= - alpha * beta / (alpha + beta)
    K = np.exp(K)

    # When it returns several values, the output becomes a list
    return p, Rp, K


# search the location of the value(object) in the list
def get_location(list_A, value):
    """
    search the location of the value(object) in the list

    input：
    list_A：搜索列表范围
    value：搜索值

    output：
    i：输出在list中的位置
    """
    for i in range(len(list_A)):
        if list_A[i] == value:
            return i


# find the integer solution of the equation: i+j+k=l
def integer_solution_num(tot_angular):
    """
    find the integer solution of the equation: i+j+k=l

    input:
    tot_angular: the number of tot_angular

    output:
    solution: a list consists of lists
    """
    solution = [[tot_angular - i - j, j, i] for i in range(0, tot_angular + 1) for j in range(0, tot_angular + 1) if
                i + j <= tot_angular]
    return solution


# find the integer solution of the equation: i+j+k=l
def integer_solution(orbital, tot_angular):
    """
    find the integer solution of the equation: i+j+k=l

    input:
    orbital:just like '2_P'
    tot_angular: a dict about the angular moment of orbitals

    output:
    solution: a list consists of lists
    """
    l = tot_angular[orbital]
    solution = [[l - i - j, j, i] for i in range(0, l + 1) for j in range(0, l + 1) if i + j <= l]
    return solution


# match the old and the new angular moment
def match_moment(list_old, new):
    """
    match the old and the new angular moment

    input:
    list_old: a list consists of lists, which are about different angular_moment
    new: a list of angular_moment

    output:
    the direction of recurrence relation
    result[0]: the old angular_moment which matches the new angular_moment
    result[1]: the direction of recurrence relation
    """
    result = []
    for sequnce, old in enumerate(list_old):
        if old[0] <= new[0] and old[1] <= new[1] and old[2] <= new[2]:
            result.append(sequnce)
            for i in range(3):
                if old[i] < new[i]:
                    result.append(i)
    return result


# the function form the coefficience product
def list_coeff_over(orb_i, orb_j, info):
    """
    the function form the coefficience product
    input:
    orb_i and orb_j: the orbitals of these atoms
    info: a dict about all kinds of information

    output: a list about the coefficience product
    """
    coeff_first = list(map(float, info['orbit_coeff'][orb_i]))
    coeff_second = list(map(float, info['orbit_coeff'][orb_j]))
    lier = [coeff_first[i] * coeff_second[j] for i in range(0, len(coeff_first))
            for j in range(0, len(coeff_second))]
    return lier


# Overlap integral between unnormalized S and S([S|S])
def SS(alpha, beta, Ra, Rb):
    """
    Overlap integral between unnormalized S and S([S|S])

    input:
        alpha: Exponential coefficient for Gaussian 1
        beta: Exponential coefficient for Gaussian 2
        R[i]: Coordinate of atom i
        R[j]: Coordinate of atom j

    output:
        S----[S|S] (unnormalized)

    source: S. Obara and A. Saika.Journal of Chemical Physics
    84.7(1986):3963-3974.
    """

    p, Rq, K = gaussian_product(alpha, beta, Ra, Rb)
    S = K
    S *= (np.pi / (alpha + beta)) ** (3 / 2)
    return S


# Overlap integal between the two unnormlized orbitals in the chosen direction
def overlapc(ac, bc, alpha, beta, Rac, Rbc, Rpc):
    """
    Overlap integal between the two unnormlized orbitals in the chosen direction

    input:
    ac: Angular momentum 1 in the chosen direction
    bc: Angular momentum 2 in the chosen direction
    alpha: Exponential coefficient for Gaussian 1
    beta: Exponential coefficient for Gaussian 2
    Rac: Coordinate of 1 in the chosen direction
    Rbc: Coordinate of 2 in the chosen direction

    output:
    Overlapc: Overlap integal between the two unnormlized orbitals in
    the chosen direction
    """
    if ac < 0 or bc < 0:
        return 0
    elif ac == 0 and bc == 0:
        return 1
    elif ac < bc:
        o = (Rpc - Rbc) * overlapc(bc - 1, ac, beta, alpha, Rbc, Rac, Rpc)
        o += (1 / (2 * (alpha + beta))) * (bc - 1) * overlapc(bc - 2, ac, beta, alpha, Rbc, Rac, Rpc)
        o += (1 / (2 * (alpha + beta))) * ac * overlapc(bc - 1, ac - 1, beta, alpha, Rbc, Rac, Rpc)
        return o
    elif ac >= bc:
        o = (Rpc - Rac) * overlapc(ac - 1, bc, alpha, beta, Rac, Rbc, Rpc)
        o += (1 / (2 * (alpha + beta))) * (ac - 1) * overlapc(ac - 2, bc, alpha, beta, Rac, Rbc, Rpc)
        o += (1 / (2 * (alpha + beta))) * bc * overlapc(ac - 1, bc - 1, alpha, beta, Rac, Rbc, Rpc)
        return o


# Overlap integal between the two unnormalized orbitals(且并未乘以初值)
def overlapall(a, b, alpha, beta, Ra, Rb, Rp):
    """
    Overlap integal between the two unnormalized orbitals

    input:
    a: Angular momentum 1
    b: Angular momentum 2
    SS: Overlap integral between unnormalized S and S([S|S])
    alpha: Exponential coefficient for Gaussian 1
    beta: Exponential coefficient for Gaussian 2
    Ra: Coordinate of 1
    Rb: Coordinate of 2
    Rp: The center of Gaussian product

    Output:
    Overlap integral
    """
    inte = overlapc(a[0], b[0], alpha, beta, Ra[0], Rb[0], Rp[0])
    inte *= overlapc(a[1], b[1], alpha, beta, Ra[1], Rb[1], Rp[1])
    inte *= overlapc(a[2], b[2], alpha, beta, Ra[2], Rb[2], Rp[2])
    return inte


# 归一化的重叠积分
def overlap(a, b, alpha, beta, Ra, Rb):
    """
    归一化的重叠积分

    input:
    a: Angular momentum 1
    b: Angular momentum 2
    alpha: Exponential coefficient for Gaussian 1
    beta: Exponential coefficient for Gaussian 2
    Ra: Coordinate of 1
    Rb: Coordinate of 2

    output:
    the Overlap integal between the two normalized orbitals
    """
    p, Rp, K = gaussian_product(alpha, beta, Ra, Rb)
    ss = SS(alpha, beta, Ra, Rb)
    inte = overlapall(a, b, alpha, beta, Ra, Rb, Rp)
    inte *= norm(a[0], a[1], a[2], alpha) * norm(b[0], b[1], b[2], beta) * ss
    return inte


# a list of overlap integral
def list_overlap(a, b, list_alpha, list_beta, Ra, Rb):
    """
    a list of overlap integral

    input:
    a: Angular momentum 1
    b: Angular momentum 2
    list_alpha: A list of exponential coefficient for Gaussian 1
    list_beta: A list of exponential coefficient for Gaussian 2
    Ra: Coordinate of 1
    Rb: Coordinate of 2

    output:
    a list of overlap integral
    """
    lt_op = [overlap(a, b, alpha, beta, Ra, Rb) for alpha in list_alpha
             for beta in list_beta]
    return lt_op


# calculate the overlap integer of two orbitals
def orb_orb_overlap(atom_i, atom_j, ang_mom_i, ang_mom_j, orb_i, orb_j, info):
    """
    calculate the overlap integer of two orbitals
    input:
    atom_i and atom_j: two atoms(just number)
    ang_mom_i and ang_mom_j: the angular moment of one orbital
    orb_i and orb_j:just like '2_P'
    info: a dict about all kinds of information

    output:
    the value of the overlap integer of two orbitals
    """
    coord_first = info[atom_i]['coordinate']
    coord_second = info[atom_j]['coordinate']
    zeta_first = list(map(float, info[atom_i]['zeta'][orb_i]))
    zeta_second = list(map(float, info[atom_j]['zeta'][orb_j]))
    coeff_product = np.array(list_coeff_over(orb_i, orb_j, info))
    overlap_product = np.array(list_overlap(ang_mom_i, ang_mom_j, zeta_first, zeta_second, coord_first, coord_second))
    result = np.dot(coeff_product, overlap_product)
    return result


# get the overlap integer matrix
def overlap_mat(counter, ele_num, orb_order, tot_angular, info):
    """
    get the overlap integer matrix

    input:
    counter: the number of oritals
    ele_num: the number of atoms
    orb_order : the order of the orbitals
    tot_angular: the angular moment of the orbitals
    info: a dict about all kinds of information

    output:
    the overlap integer matrix
    """
    orbital_counter = -1
    overlap_arr = np.zeros((counter, counter))
    for atom_i in range(1, ele_num + 1):
        for orbital_i in orb_order:
            if orbital_i in info[atom_i]['zeta'].keys():
                for angular_moment_i in integer_solution(orbital_i, tot_angular):
                    orbital_counter += 1
                    orbital_counter_0 = 0
                    for atom_j in range(1, atom_i + 1):
                        if atom_j == atom_i:
                            threshold = get_location(orb_order, orbital_i)
                        else:
                            threshold = len(orb_order) - 1
                        for orbital_j in orb_order[0: threshold + 1]:
                            if atom_j == atom_i and orbital_j == orbital_i:
                                threshold_orb = get_location(integer_solution(orbital_j, tot_angular), angular_moment_i)
                            else:
                                threshold_orb = len(integer_solution(orbital_j, tot_angular)) - 1
                            if orbital_j in info[atom_j]['zeta'].keys():
                                for angular_moment_j in integer_solution(orbital_j, tot_angular)[0: threshold_orb + 1]:
                                    overlap_arr[orbital_counter][orbital_counter_0] = orb_orb_overlap(
                                        atom_i, atom_j, angular_moment_i, angular_moment_j, orbital_i, orbital_j, info)
                                    orbital_counter_0 += 1
                            else:
                                break
            else:
                break
    for orb_i in range(0, len(overlap_arr[counter - 1])):
        for orb_j in range(orb_i + 1, len(overlap_arr[counter - 1])):
            overlap_arr[orb_i][orb_j] = overlap_arr[orb_j][orb_i]
    return overlap_arr


# ([S|K|S]/[S|S])
def KK(alpha, beta, Ra, Rb):
    """
    ([S|K|S]/[S|S])

    Input:
    alpha: Angular momentum 1
    beta: Angular momentum 2
    Ra: Coordinate of the atom 1
    Rb: Coordinate of the atom 2

    Output:
    [S|K|S]/[S|S]
    """

    t = (alpha * beta) / (alpha + beta)
    r2 = ((Ra[0] - Rb[0]) ** 2) + ((Ra[1] - Rb[1]) ** 2) + ((Ra[2] - Rb[2]) ** 2)
    kk = t * (3 - 2 * t * r2)
    return kk


# Kinetic integal between the two unnormlized orbitals.This is just the coefficient of the kinetic integral,
# it needs to multiply [S|S] later.
def kineticc(a0, a1, a2, b0, b1, b2, alpha, beta, Ra, Rb, Rp, kk):
    """
    Kinetic integal between the two unnormlized orbitals
    This is just the coefficient of the kinetic integral,
    it needs to multiply [S|S] later.

    Input:
    a0: Angular momentum 1 in the direction 1
    a1: Angular momentum 1 in the direction 2
    a2: Angular momentum 1 in the direction 3
    b0: Angular momentum 2 in the direction 1
    b1: Angular momentum 2 in the direction 2
    b2: Angular momentum 2 in the direction 3
    alpha: Exponential coefficient for Gaussian 1
    beta: Exponential coefficient for Gaussian 2
    Ra: Coordinate of atom 1
    Rb: Coordinate of atom 2
    Rp: The center of Gaussian product
    kk: [S|K|S]/[S|S]

    Output:
    Kinetic integal between the two unnormlized orbitals/[S|S]
    """

    t = (alpha * beta) / (alpha + beta)
    if a0 < 0 or a1 < 0 or a2 < 0 or b0 < 0 or b1 < 0 or b2 < 0:
        return 0
    elif a0 == 0 and a1 == 0 and a2 == 0 and b0 == 0 and b1 == 0 and b2 == 0:
        return kk
    elif 0 <= a0 < b0 and b0 >= 0:
        inte = (Rp[0] - Rb[0]) * kineticc(b0 - 1, a1, a2, a0, b1, b2, beta, alpha, Ra, Rb, Rp, kk)
        inte += (1 / (2 * (alpha + beta))) * (b0 - 1) * (kineticc(b0 - 2, a1, a2, a0, b1, b2, beta, alpha, Ra, Rb, Rp,
                                                                  kk))
        inte += (1 / (2 * (alpha + beta))) * a0 * kineticc(b0 - 1, a1, a2, a0 - 1, b1, b2, beta, alpha, Ra, Rb, Rp, kk)
        c0 = np.array([b0, a1, a2])
        c1 = np.array([a0, b1, b2])
        c2 = np.array([b0 - 2, a1, a2])
        inte += 2 * t * (
                overlapall(c0, c1, beta, alpha, Rb, Ra, Rp) - (1 / (2 * beta)) * (b0 - 1) * overlapall(c2, c1, beta,
                                                                                                       alpha, Rb,
                                                                                                       Ra, Rp))
        return inte
    elif a0 >= 0 and 0 <= b0 <= a0 and a0 > 0:
        inte = (Rp[0] - Ra[0]) * kineticc(a0 - 1, a1, a2, b0, b1, b2, alpha, beta, Ra, Rb, Rp, kk)
        inte += (1 / (2 * (alpha + beta))) * (a0 - 1) * kineticc(a0 - 2, a1, a2, b0, b1, b2, alpha, beta, Ra, Rb, Rp,
                                                                 kk)
        inte += (1 / (2 * (alpha + beta))) * b0 * kineticc(a0 - 1, a1, a2, b0 - 1, b1, b2, alpha, beta, Ra, Rb, Rp, kk)
        c0 = np.array([a0, a1, a2])
        c1 = np.array([b0, b1, b2])
        c2 = np.array([a0 - 2, a1, a2])
        inte += 2 * t * (
                overlapall(c0, c1, alpha, beta, Ra, Rb, Rp)
                - (1 / (2 * alpha)) * (a0 - 1) * overlapall(c2, c1, alpha, beta, Rb, Ra, Rp)
        )
        return inte
    elif a0 == 0 and b0 == 0 and a1 >= 0 and b1 >= 0 and a1 < b1:
        inte = (Rp[1] - Rb[1]) * kineticc(a0, b1 - 1, a2, b0, a1, b2, beta, alpha, Ra, Rb, Rp, kk)
        inte += (1 / (2 * (alpha + beta))) * (b1 - 1) * kineticc(a0, b1 - 2, a2, b0, a1, b2, beta, alpha, Ra, Rb, Rp,
                                                                 kk)
        inte += (1 / (2 * (alpha + beta))) * a1 * kineticc(a0, b1 - 1, a2, b0, a1 - 1, b2, beta, alpha, Ra, Rb, Rp, kk)
        c0 = np.array([a0, b1, a2])
        c1 = np.array([b0, a1, b2])
        c2 = np.array([a0, b1 - 2, a2])
        inte += 2 * t * (
                overlapall(c0, c1, beta, alpha, Rb, Ra, Rp) - (1 / (2 * beta)) * (b1 - 1) * overlapall(c2, c1, beta,
                                                                                                       alpha, Rb,
                                                                                                       Ra, Rp))
        return inte
    elif a0 == 0 and b0 == 0 and a1 >= 0 and 0 <= b1 <= a1 and a1 > 0:
        inte = (Rp[1] - Ra[1]) * kineticc(a0, a1 - 1, a2, b0, b1, b2, alpha, beta, Ra, Rb, Rp, kk)
        inte += (1 / (2 * (alpha + beta))) * (a1 - 1) * kineticc(a0, a1 - 2, a2, b0, b1, b2, alpha, beta, Ra, Rb, Rp,
                                                                 kk)
        inte += (1 / (2 * (alpha + beta))) * b1 * kineticc(a0, a1 - 1, a2, b0, b1 - 1, b2, alpha, beta, Ra, Rb, Rp, kk)
        c0 = np.array([a0, a1, a2])
        c1 = np.array([b0, b1, b2])
        c2 = np.array([a0, a1 - 2, a2])
        inte += 2 * t * (
                overlapall(c0, c1, alpha, beta, Ra, Rb, Rp)
                - (1 / (2 * alpha)) * (a1 - 1) * overlapall(c2, c1, alpha, beta, Ra, Rb, Rp)
        )
        return inte
    elif a0 == 0 and b0 == 0 and a1 == 0 and b1 == 0 and a2 >= 0 and b2 >= 0 and a2 < b2:
        inte = (Rp[2] - Rb[2]) * kineticc(a0, a1, b2 - 1, b0, b1, a2, beta, alpha, Ra, Rb, Rp, kk)
        inte += (1 / (2 * (alpha + beta))) * (b2 - 1) * kineticc(a0, a1, b2 - 2, b0, b1, a2, beta, alpha, Ra, Rb, Rp,
                                                                 kk)
        inte += (1 / (2 * (alpha + beta))) * a2 * kineticc(a0, a1, b2 - 1, b0, b1, a2 - 1, beta, alpha, Ra, Rb, Rp, kk)
        c0 = np.array([a0, a1, b2])
        c1 = np.array([b0, b1, a2])
        c2 = np.array([a0, a1, b2 - 2])
        inte += 2 * t * (
                overlapall(c0, c1, beta, alpha, Rb, Ra, Rp) - (1 / (2 * beta)) * (b2 - 1) * overlapall(c2, c1, beta,
                                                                                                       alpha, Rb,
                                                                                                       Ra, Rp))
        return inte
    elif a0 == 0 and b0 == 0 and a1 == 0 and b1 == 0 and a2 >= 0 and 0 <= b2 <= a2 and a2 > 0:
        inte = (Rp[2] - Ra[2]) * kineticc(a0, a1, a2 - 1, b0, b1, b2, alpha, beta, Ra, Rb, Rp, kk)
        inte += (1 / (2 * (alpha + beta))) * (a2 - 1) * kineticc(a0, a1, a2 - 2, b0, b1, b2, alpha, beta, Ra, Rb, Rp,
                                                                 kk)
        inte += (1 / (2 * (alpha + beta))) * b2 * kineticc(a0, a1, a2 - 1, b0, b1, b2 - 1, alpha, beta, Ra, Rb, Rp, kk)
        c0 = np.array([a0, a1, a2])
        c1 = np.array([b0, b1, b2])
        c2 = np.array([a0, a1, a2 - 2])
        inte += 2 * t * (
                overlapall(c0, c1, alpha, beta, Ra, Rb, Rp)
                - (1 / (2 * alpha)) * (a2 - 1) * overlapall(c2, c1, alpha, beta, Ra, Rb, Rp)
        )
        return inte


# Kinetic integral between two normalized gaussian
def kinetic(a, b, ss, alpha, beta, Ra, Rb):
    """
    Kinetic integral between two normalized gaussian

    Input:
    a: Angular momentum 1
    b: Angular momentum 2
    SS: Overlap integral between unnormalized S and S([S|S])
    alpha: Exponential coefficient for Gaussian 1
    beta: Exponential coefficient for Gaussian 2
    Ra: Coordinate of 1
    Rb: Coordinate of 2
    Rp: The center of Gaussian product

    Output:
    Kinetic integral between two normalized gaussian
    """
    p, Rp, K = gaussian_product(alpha, beta, Ra, Rb)
    kk = KK(alpha, beta, Ra, Rb)
    K = kineticc(a[0], a[1], a[2], b[0], b[1], b[2], alpha, beta, Ra, Rb, Rp, kk)
    K *= norm(a[0], a[1], a[2], alpha) * norm(b[0], b[1], b[2], beta) * ss
    return K


# a list of kinetic integral
def list_kinetic(a, b, list_alpha, list_beta, Ra, Rb):
    """
    a list of kinetic integral

    input:
    a: Angular momentum 1
    b: Angular momentum 2
    list_alpha: A list of exponential coefficient for Gaussian 1
    list_beta: A list of exponential coefficient for Gaussian 2
    Ra: Coordinate of 1
    Rb: Coordinate of 2

    output:
    a list of kinetic integral
    """
    lt_kc = [kinetic(a, b, SS(alpha, beta, Ra, Rb), alpha, beta, Ra, Rb) for alpha in list_alpha
             for beta in list_beta]
    return lt_kc


# calculate the kinetic integer of two orbitals
def orb_orb_kinetic(atom_i, atom_j, ang_mom_i, ang_mom_j, orb_i, orb_j, info):
    """
    calculate the kinetic integer of two orbitals

    input:
    atom_i and atom_j: two atoms(just number)
    ang_mom_i and ang_mom_j: the angular moment of one orbital
    orb_i and orb_j:just like '2_P'
    info: a dict about all kinds of information

    output:
    the value of the kinetic integer of two orbitals
    """
    coord_first = info[atom_i]['coordinate']
    coord_second = info[atom_j]['coordinate']
    zeta_first = list(map(float, info[atom_i]['zeta'][orb_i]))
    zeta_second = list(map(float, info[atom_j]['zeta'][orb_j]))
    coeff_product = np.array(list_coeff_over(orb_i, orb_j, info))
    kinetic_product = np.array(list_kinetic(ang_mom_i, ang_mom_j, zeta_first, zeta_second, coord_first, coord_second))
    result = np.dot(coeff_product, kinetic_product)
    return result


# get the kinetic integer matrix
def kinetic_mat(counter, ele_num, orb_order, tot_angular, info):
    """
    get the kinetic integer matrix

    input:
    counter: the number of oritals
    ele_num: the number of atoms
    orb_order : the order of the orbitals
    tot_angular: the angular moment of the orbitals
    info: a dict about all kinds of information

    output:
    the kinetic integer matrix
    """
    kinetic_arr = np.zeros((counter, counter))
    orbital_counter = -1
    for atom_i in range(1, ele_num + 1):
        for orbital_i in orb_order:
            if orbital_i in info[atom_i]['zeta'].keys():
                for angular_moment_i in integer_solution(orbital_i, tot_angular):
                    orbital_counter += 1
                    orbital_counter_0 = 0
                    for atom_j in range(1, atom_i + 1):
                        if atom_j == atom_i:
                            threshold = get_location(orb_order, orbital_i)
                        else:
                            threshold = len(orb_order) - 1
                        for orbital_j in orb_order[0: threshold + 1]:
                            if atom_j == atom_i and orbital_j == orbital_i:
                                threshold_orb = get_location(integer_solution(orbital_j, tot_angular), angular_moment_i)
                            else:
                                threshold_orb = len(integer_solution(orbital_j, tot_angular)) - 1
                            if orbital_j in info[atom_j]['zeta'].keys():
                                for angular_moment_j in integer_solution(orbital_j, tot_angular)[0: threshold_orb + 1]:
                                    kinetic_arr[orbital_counter][orbital_counter_0] = orb_orb_kinetic(
                                        atom_i, atom_j, angular_moment_i, angular_moment_j, orbital_i, orbital_j, info)
                                    orbital_counter_0 += 1
                            else:
                                break
            else:
                break
    for orb_i in range(0, len(kinetic_arr[counter - 1])):
        for orb_j in range(orb_i + 1, len(kinetic_arr[counter - 1])):
            kinetic_arr[orb_i][orb_j] = kinetic_arr[orb_j][orb_i]
    return kinetic_arr


# Boys function array.
def boys_arr(m, w):
    """
    Boys function array.

    INPUT:
        m: Boys function index
        w: Boys function variable

    OUTPUT:
        F: An array of values of the Boys function for index i from 0 to m evaluated at w.

    Source:
        Evaluation of the Boys Function using Analytical Relations
        I. I. Guseinov and B. A. Mamedov
        Journal of Mathematical Chemistry
        2006
    """
    F = np.zeros(m + 1)
    if w < 10 ** (-14):
        for i in range(m + 1):
            F[i] = 1.0 / (2.0 * float(i) + 1.0)
    elif 10 ** (-14) < w < float(m) + 1.5:
        x = 1.
        s = 1.
        b = float(m) + 0.5
        e = 0.5 * mp.exp(-w)
        for i in range(1, 10000):
            x = x * w / (b + i)
            s = s + x
            if x < 10 ** (-14):
                break
        F[m] = e * s / b
        if m > 0:
            for i in range(m, 0, -1):
                b = b - 1.
                F[i - 1] = (e + w * F[i]) / b
    else:
        t = np.sqrt(w)
        e = mp.exp(-w)
        F[0] = 0.88622692545275801 * mp.erf(t) / t
        if m > 0:
            for i in range(1, m + 1):
                F[i] = ((2. * i - 1.) * F[i - 1] - e) / (2 * w)
    return F


# (S|V|S)_n
def nuclear_00(p, KAB, boy_array, tot_ang):
    """
    (S|V|S)_n

    input:
    p:Exponential coefficients of Gaussian product
    KAB:Gaussian product coefficient
    boy_array:an array of boys functions for different n
    tot_ang:the sum of all angular moments

    Output:
    (S|V|S)_n
    """
    coeff = (2*np.pi / p) * KAB
    os_00 = np.zeros(tot_ang + 1)
    for i in range(tot_ang + 1):
        os_00[i] = coeff * boy_array[i]
    return os_00


# a dict about (i|V|0) (i从0到tot_ang) 未归一化
def nuclear_i0(tot_ang, os_00, Rp, Rc, Ra, p):
    """
    the first step of OS method

    Input:
    tot_ang:the sum of all angular moments
    os_00:(S|V|S)_n
    Rp: the center of Gaussian product
    Rc: the coordinate of atom C
    Ra: the centers of Gaussian 1
    p: Exponential coefficient of Gaussian product

    intermediate
    output:os_i0

    OS: {0:{0:***},1:{0:***, 1:***,....}....}
    """
    OS = {}
    auxi = {}
    for total_ang in range(0, tot_ang + 1):
        auxi.setdefault(total_ang, {})
        if total_ang > 0:
            old_list = integer_solution_num(total_ang - 1)
        for sequence, ang_mom in enumerate(integer_solution_num(total_ang)):  # start from 0
            auxi[total_ang].setdefault(sequence, {})
            if total_ang == 0:
                for index in range(tot_ang + 1 - total_ang):
                    auxi[total_ang][sequence][index] = os_00[index]
            else:
                for index in range(tot_ang + 1 - total_ang):
                    direction = match_moment(old_list, ang_mom)
                    auxi[total_ang][sequence][index] = (Rp[direction[1]] - Ra[direction[1]]) * \
                                                       auxi[total_ang - 1][direction[0]][index]
                    auxi[total_ang][sequence][index] -= (Rp[direction[1]] - Rc[direction[1]]) * \
                                                        auxi[total_ang - 1][direction[0]][index + 1]
                    if old_list[direction[0]][direction[1]] > 0:
                        location_list = old_list[direction[0]][:]
                        location_list[direction[1]] -= 1
                        location = get_location(integer_solution_num(total_ang - 2), location_list)
                        auxi[total_ang][sequence][index] += (old_list[direction[0]][direction[1]] / (2 * p)) * (
                                auxi[total_ang - 2][location][index] - auxi[total_ang - 2][location][index + 1])
    for total_ang in range(0, tot_ang + 1):
        OS.setdefault(total_ang, {})
        for sequence, ang_mom in enumerate(integer_solution_num(total_ang)):  # start from 0
            OS[total_ang].setdefault(sequence, {})
            OS[total_ang][sequence] = auxi[total_ang][sequence][0]
    return OS


# a dict about (i|V|j) (i从0到a，j从0到b)
def nuclear_ij(OS_i0, a, b, Ra, Rb):
    """
    a dict about (i|j) (i从0到a，j从0到b)

    input:
    OS_i0: a dict about (i|0)
    a and b: a 和 b 是两个轨道对应的角动量的值。
    Ra: Coordinate of 1
    Rb: Coordinate of 2

    output:
    a dict about (i|j) (i从0到a，j从0到b)
    """
    auxi = {}
    for PAM_2 in range(0, b + 1):
        auxi.setdefault(PAM_2, {})
        if PAM_2 == 0:
            for PAM_1 in range(a, a + b + 1 - PAM_2):
                auxi[PAM_2].setdefault(PAM_1, {})
                for sequence_2, ang_mom_2 in enumerate(integer_solution_num(PAM_2)):
                    auxi[PAM_2][PAM_1].setdefault(sequence_2, {})
                    for sequence_1, ang_mom_1 in enumerate(integer_solution_num(PAM_1)):
                        auxi[0][PAM_1][0][sequence_1] = OS_i0[PAM_1][sequence_1]
        else:
            old_list = integer_solution_num(PAM_2 - 1)
            for PAM_1 in range(a, a + b + 1 - PAM_2):
                auxi[PAM_2].setdefault(PAM_1, {})
                for sequence_2, ang_mom_2 in enumerate(integer_solution_num(PAM_2)):
                    auxi[PAM_2][PAM_1].setdefault(sequence_2, {})
                    for sequence_1, ang_mom_1 in enumerate(integer_solution_num(PAM_1)):
                        direction = match_moment(old_list, ang_mom_2)
                        auxi[PAM_2][PAM_1][sequence_2][sequence_1] = (Ra[direction[1]] - Rb[direction[1]])
                        auxi[PAM_2][PAM_1][sequence_2][sequence_1] *= auxi[PAM_2 - 1][PAM_1][direction[0]][sequence_1]
                        location_list_1_up = ang_mom_1[:]
                        location_list_1_up[direction[1]] += 1
                        location_1_up = get_location(integer_solution_num(PAM_1 + 1), location_list_1_up)
                        auxi[PAM_2][PAM_1][sequence_2][sequence_1] += auxi[PAM_2 - 1][PAM_1 + 1][direction[0]][
                            location_1_up]
    os_ab = {}
    for sequence_1, ang_mom_1 in enumerate(integer_solution_num(a)):
        os_ab.setdefault(sequence_1, {})
        for sequence_2, ang_mom_2 in enumerate(integer_solution_num(b)):
            os_ab[sequence_1][sequence_2] = auxi[b][a][sequence_2][sequence_1]
    return os_ab


# 一个由两个特定角量子数不同的磁量子数构成的归一化的核势能积分的字典
def norm_nuclear(Ra, Rb, Rc, alpha, beta, orb_i, orb_j, tot_angular):
    """
    一个由两个特定角量子数不同的磁量子数构成的归一化的核势能积分的字典

    input:
    Ra: Coordinate of atom 1
    Rb: Coordinate of atom 2
    Rc: Coordinate of atom 3
    alpha: Exponential coefficient for Gaussian 1
    beta: Exponential coefficient for Gaussian 2
    orb_i and orb_j: the orbitals of these atoms
    tot_angular: a dict of angular moment of the orbitals

    output:
    一个由两个特定角量子数不同的磁量子数构成的归一化的核势能积分的字典
    """
    a = tot_angular[orb_i]
    b = tot_angular[orb_j]
    tot_ang = a+b
    ei_ab = {}
    p, Rp, KAB = gaussian_product(alpha, beta, Ra, Rb)
    boys_array = boys_arr(tot_ang, p*np.dot(Rp-Rc, Rp-Rc))
    os_00 = nuclear_00(p, KAB, boys_array, tot_ang)
    os_i0 = nuclear_i0(tot_ang, os_00, Rp, Rc, Ra, p)
    os_ab = nuclear_ij(os_i0, a, b, Ra, Rb)
    for sequence_1, ang_mom_1 in enumerate(integer_solution_num(a)):
        ei_ab.setdefault(sequence_1, {})
        for sequence_2, ang_mom_2 in enumerate(integer_solution_num(b)):
            ei_ab[sequence_1][sequence_2] = os_ab[sequence_1][sequence_2]
            ei_ab[sequence_1][sequence_2] *= norm(ang_mom_1[0], ang_mom_1[1], ang_mom_1[2], alpha)
            ei_ab[sequence_1][sequence_2] *= norm(ang_mom_2[0], ang_mom_2[1], ang_mom_2[2], beta)
    return ei_ab


# a list about [i|V|j]
def list_nuclear(list_alpha, list_beta, Ra, Rb, Rc, orb_i, orb_j, tot_angular):
    """
    a list about [a|V|b]

    Input:
    a: Angular momentum 1
    b: Angular momentum 2
    list_alpha: Exponential coefficients for Gaussians about an orbital(nlm)
    list_beta: Exponential coefficients for Gaussians about an orbital(nlm)
    Ra: Coordinate of 1
    Rb: Coordinate of 2
    Rc: Coordinate of the charged nuclear
    Rp: The center of Gaussian product
    ss: Overlap integral between unnormalized S and S([S|S])

    Output:
    a list about [a|V|b]
    """
    lt_nr = [norm_nuclear(Ra, Rb, Rc, alpha, beta, orb_i, orb_j, tot_angular)
             for alpha in list_alpha for beta in list_beta]
    return lt_nr


# calculate the nuclear integer of two orbitals
def orb_orb_nuclear(atom_i, atom_j, atom_k, orb_i, orb_j, info, tot_angular):
    """
    calculate the nuclear integer of two orbitals

    input:
    atom_i and atom_j: two atoms(just number)
    atom_k: a reference charged atom
    orb_i and orb_j:just like '2_P'
    info: a dict about all kinds of information

    output:
    the value of the nuclear integer of two orbitals
    """
    a = tot_angular[orb_i]
    b = tot_angular[orb_j]
    coord_first = info[atom_i]['coordinate']
    coord_second = info[atom_j]['coordinate']
    coord_third = info[atom_k]['coordinate']
    zeta_first = list(map(float, info[atom_i]['zeta'][orb_i]))
    zeta_second = list(map(float, info[atom_j]['zeta'][orb_j]))
    coeff_product = list_coeff_over(orb_i, orb_j, info)
    nuclear_product = list_nuclear(zeta_first, zeta_second, coord_first, coord_second, coord_third, orb_i, orb_j,
                                   tot_angular)
    NC = info[atom_k]['atomic_number']
    ei = {}
    for sequence_1, ang_mom_1 in enumerate(integer_solution_num(a)):
        ei.setdefault(sequence_1,{})
        for sequence_2, ang_mom_2 in enumerate(integer_solution_num(b)):
            ei[sequence_1][sequence_2] = 0
            for i in range(len(coeff_product)):
                ei[sequence_1][sequence_2] += -NC*coeff_product[i]*nuclear_product[i][sequence_1][sequence_2]
    return ei


# get the kinetic integer matrix
def nuclear_mat(counter, ele_num, orb_order, tot_angular, info):
    """
    get the kinetic integer matrix

    input:
    counter: the number of oritals
    ele_num: the number of atoms
    orb_order : the order of the orbitals
    tot_angular: the angular moment of the orbitals
    info: a dict about all kinds of information

    output:
    a dict about nuclear integer matrixs
    """
    nuclear_dict = {}
    for atom_k in range(1, ele_num + 1):
        orbital_counter_1 = 0
        nuclear_arr = np.zeros((counter, counter))
        for atom_i in range(1, ele_num + 1):
            for orbital_i in orb_order:
                if orbital_i in info[atom_i]['zeta'].keys():
                    orbital_counter_2 = 0
                    for atom_j in range(1, atom_i + 1):
                        if atom_j == atom_i:
                            threshold = get_location(orb_order, orbital_i)
                        else:
                            threshold = len(orb_order)-1
                        for orbital_j in orb_order[0: threshold+1]:
                            if orbital_j in info[atom_j]['zeta'].keys():
                                ei_nucl = orb_orb_nuclear(atom_i, atom_j, atom_k, orbital_i, orbital_j, info, tot_angular)
                                for counter_1 in range(orbital_counter_1, orbital_counter_1+len(integer_solution(orbital_i, tot_angular))):
                                    for counter_2 in range(orbital_counter_2, orbital_counter_2+len(integer_solution(orbital_j, tot_angular))):
                                        nuclear_arr[counter_1][counter_2] = ei_nucl[counter_1 - orbital_counter_1][counter_2 - orbital_counter_2]
                                orbital_counter_2 += len(integer_solution(orbital_j, tot_angular))
                            else:
                                break
                    orbital_counter_1 += len(integer_solution(orbital_i, tot_angular))
                else:
                    break
        for orb_i in range(counter):
            for orb_j in range(orb_i+1, counter):
                nuclear_arr[orb_i][orb_j] = nuclear_arr[orb_j][orb_i]
        nuclear_dict.setdefault(atom_k, nuclear_arr)
    return nuclear_dict


# an array of (SS|SS)_n(four unnormalized gaussian function)
def SSSS_n(p, q, KAB, KCD, boy_array, tot_ang):
    """
    (SS|SS)_n(four unnormalized gaussian function)

    Input:
    p and q: Exponential coefficients of Gaussian product
    KAB abd KCD: Gaussian product coefficients
    tot_ang:the sum of all angular moments
    boy_array:an array of boys functions for different n

    Output:(SS|SS)_n(four unnormalized gaussian function)
    """
    '''
    p, Rp, KAB = gaussian_product(alpha, beta, Ra, Rb)
    q, Rq, KCD = gaussian_product(gamma, delta, Rc, Rd)
    u = p * q / (p + q)
    boy_array = boy_recurrence(tot_ang, u * np.dot(Rp - Rq, Rp - Rq))
    '''
    coeff = 2 * (np.pi ** 2.5) / (p * q * np.sqrt(p + q))
    coeff *= KAB * KCD
    SSSS = np.zeros(tot_ang + 1)
    for i in range(tot_ang + 1):
        SSSS[i] = coeff * boy_array[i]
    return SSSS


# a dict about (i0|00) (i从0到tot_ang) 未归一化
def OS_i000(tot_ang, SSSS, Rp, Rq, Ra, u, p):
    """
    the first step of OS method

    Input:
    tot_ang:the sum of all angular moments
    SSSS:four unnormalized gaussian function electronic integrals
    Rp and Rq: the center of Gaussian product
    Ra: the centers of Gaussian 1
    u: p*q/(p+q)
    p: Exponential coefficient of Gaussian product

    intermediate
    auxi:{0:{0:{0:***,1:***,2:***}1:{0:***.1:***}},1:{}} {total_angular:{the sequence of different angular:{index: }}}
    output:OS_i000

    OS: {0:{0:***},1:{0:***, 1:***,....}....}
    """
    OS = {}
    auxi = {}
    for total_ang in range(0, tot_ang + 1):
        auxi.setdefault(total_ang, {})
        if total_ang > 0:
            old_list = integer_solution_num(total_ang - 1)
        for sequence, ang_mom in enumerate(integer_solution_num(total_ang)):  # start from 0
            auxi[total_ang].setdefault(sequence, {})
            if total_ang == 0:
                for index in range(tot_ang + 1 - total_ang):
                    auxi[total_ang][sequence].setdefault(index, SSSS[index])
            else:
                for index in range(tot_ang + 1 - total_ang):
                    direction = match_moment(old_list, ang_mom)
                    auxi[total_ang][sequence][index] = (Rp[direction[1]] - Ra[direction[1]]) * \
                                                       auxi[total_ang - 1][direction[0]][index]
                    auxi[total_ang][sequence][index] -= (u / p) * (Rp[direction[1]] - Rq[direction[1]]) * \
                                                        auxi[total_ang - 1][direction[0]][index + 1]
                    if old_list[direction[0]][direction[1]] > 0:
                        location_list = old_list[direction[0]][:]
                        location_list[direction[1]] -= 1
                        location = get_location(integer_solution_num(total_ang - 2), location_list)
                        auxi[total_ang][sequence][index] += (old_list[direction[0]][direction[1]] / (2 * p)) * (
                                auxi[total_ang - 2][location][index] - (u / p) * auxi[total_ang - 2][location][
                            index + 1]
                        )
    for total_ang in range(0, tot_ang + 1):
        OS.setdefault(total_ang, {})
        for sequence, ang_mom in enumerate(integer_solution_num(total_ang)):  # start from 0
            OS[total_ang].setdefault(sequence, {})
            OS[total_ang][sequence] = auxi[total_ang][sequence][0]
    return OS


#     electron transfer recurrence relation
#     (i, 0, 0, 0) ---- (i, 0, k, 0)
# a dict about (i0|k0) (i从0到a+b)(k从0到c+d)  未归一化
def electron_transfer_RR(OS, beta, delta, Ra, Rb, Rc, Rd, p, q, a, b, c, d):
    """
    electron transfer recurrence relation
    (i, 0, 0, 0) ---- (i, 0, k, 0)

    Input:
    OS:OS_i000
    beta: Exponential coefficient for Gaussian 2
    delta: Exponential coefficient for Gaussian 4
    Ra: Coordinate of 1
    Rb: Coordinate of 2
    Rc: Coordinate of 3
    Rd: Coordinate of 4
    p and q: Exponential coefficient of Gaussian product
    a, b, c, d:the total angular moment of cartesian GTO

    Output:
    (i, 0, k, 0)
    ETR:{0:***{0:***
    """
    RR = {}
    for PAM_2 in range(0, c + d + 1):  # PAM_2 is partial angular moment 2
        RR.setdefault(PAM_2, {})
        if PAM_2 == 0:
            for PAM_1 in range(max(0, PAM_2 + a - (c + d)),
                               a + b + c + d + 1 - PAM_2):  # PAM_1 is partial angular moment 1
                RR[PAM_2].setdefault(PAM_1, {})
                for sequence_2, ang_mom_2 in enumerate(integer_solution_num(PAM_2)):
                    RR[PAM_2][PAM_1].setdefault(sequence_2, {})
                    for sequence_1, ang_mom_1 in enumerate(integer_solution_num(PAM_1)):
                        RR[PAM_2][PAM_1][sequence_2].setdefault(sequence_1, OS[PAM_1][sequence_1])
        else:
            old_list_2 = integer_solution_num(PAM_2 - 1)  # 0,0,0
            for PAM_1 in range(max(0, PAM_2 + a - (c + d)),
                               a + b + c + d + 1 - PAM_2):  # PAM_1 is partial angular moment 1
                RR[PAM_2].setdefault(PAM_1, {})
                for sequence_2, ang_mom_2 in enumerate(integer_solution_num(PAM_2)):  # 0
                    RR[PAM_2][PAM_1].setdefault(sequence_2, {})
                    for sequence_1, ang_mom_1 in enumerate(integer_solution_num(PAM_1)):  # 0
                        direction = match_moment(old_list_2, ang_mom_2)  # direction matches the angular moment 2
                        if ang_mom_2[direction[1]] == 1:
                            if ang_mom_1[direction[1]] == 0:
                                RR[PAM_2][PAM_1][sequence_2][sequence_1] = -(
                                        beta * (Ra[direction[1]] - Rb[direction[1]]) + delta * (
                                        Rc[direction[1]] - Rd[direction[1]])) / q
                                RR[PAM_2][PAM_1][sequence_2][sequence_1] *= RR[PAM_2 - 1][PAM_1][direction[0]][
                                    sequence_1]
                                location_list_1_up = ang_mom_1[:]  # search where i+1 in sequnce_1
                                location_list_1_up[direction[1]] += 1
                                location_1_up = get_location(integer_solution_num(PAM_1 + 1), location_list_1_up)
                                RR[PAM_2][PAM_1][sequence_2][sequence_1] -= (p / q) * \
                                                                            RR[PAM_2 - 1][PAM_1 + 1][direction[0]][
                                                                                location_1_up]
                            else:
                                RR[PAM_2][PAM_1][sequence_2][sequence_1] = -(
                                        beta * (Ra[direction[1]] - Rb[direction[1]]) + delta * (
                                        Rc[direction[1]] - Rd[direction[1]])) / q
                                RR[PAM_2][PAM_1][sequence_2][sequence_1] *= RR[PAM_2 - 1][PAM_1][direction[0]][
                                    sequence_1]
                                location_list_1_down = ang_mom_1[:]  # search where i-1 in sequnce_1
                                location_list_1_down[direction[1]] -= 1
                                location_1_down = get_location(integer_solution_num(PAM_1 - 1), location_list_1_down)
                                ang_dir_1 = ang_mom_1[direction[1]]  # angular moment 1 in a certain direction
                                RR[PAM_2][PAM_1][sequence_2][sequence_1] += (ang_dir_1 / (2 * q)) * \
                                                                            RR[PAM_2 - 1][PAM_1 - 1][direction[0]][
                                                                                location_1_down]
                                location_list_1_up = ang_mom_1[:]  # search where i+1 in sequnce_1
                                location_list_1_up[direction[1]] += 1
                                location_1_up = get_location(integer_solution_num(PAM_1 + 1), location_list_1_up)
                                RR[PAM_2][PAM_1][sequence_2][sequence_1] -= (p / q) * \
                                                                            RR[PAM_2 - 1][PAM_1 + 1][direction[0]][
                                                                                location_1_up]
                        else:
                            if ang_mom_1[direction[1]] == 0:
                                # first
                                RR[PAM_2][PAM_1][sequence_2][sequence_1] = -(
                                        beta * (Ra[direction[1]] - Rb[direction[1]]) + delta * (
                                        Rc[direction[1]] - Rd[direction[1]])) / q
                                RR[PAM_2][PAM_1][sequence_2][sequence_1] *= RR[PAM_2 - 1][PAM_1][direction[0]][
                                    sequence_1]
                                # third
                                location_list_2_down = ang_mom_2[:]
                                location_list_2_down[direction[1]] -= 2
                                location_2_down = get_location(integer_solution_num(PAM_2 - 2), location_list_2_down)
                                ang_dir_2 = ang_mom_2[direction[1]] - 1
                                RR[PAM_2][PAM_1][sequence_2][sequence_1] += (ang_dir_2 / (2 * q)) * \
                                                                            RR[PAM_2 - 2][PAM_1][location_2_down][
                                                                                sequence_1]
                                # forth
                                location_list_1_up = ang_mom_1[:]  # search where i+1 in sequnce_1
                                location_list_1_up[direction[1]] += 1
                                location_1_up = get_location(integer_solution_num(PAM_1 + 1), location_list_1_up)
                                RR[PAM_2][PAM_1][sequence_2][sequence_1] -= (p / q) * \
                                                                            RR[PAM_2 - 1][PAM_1 + 1][direction[0]][
                                                                                location_1_up]
                            else:
                                # first
                                RR[PAM_2][PAM_1][sequence_2][sequence_1] = -(
                                        beta * (Ra[direction[1]] - Rb[direction[1]]) + delta * (
                                        Rc[direction[1]] - Rd[direction[1]])) / q
                                RR[PAM_2][PAM_1][sequence_2][sequence_1] *= RR[PAM_2 - 1][PAM_1][direction[0]][
                                    sequence_1]
                                # second
                                location_list_1_down = ang_mom_1[:]  # search where i-1 in sequnce_1
                                location_list_1_down[direction[1]] -= 1
                                location_1_down = get_location(integer_solution_num(PAM_1 - 1), location_list_1_down)
                                ang_dir_1 = ang_mom_1[direction[1]]  # angular moment 1 in a certain direction
                                RR[PAM_2][PAM_1][sequence_2][sequence_1] += (ang_dir_1 / (2 * q)) * \
                                                                            RR[PAM_2 - 1][PAM_1 - 1][direction[0]][
                                                                                location_1_down]
                                # third
                                location_list_2_down = ang_mom_2[:]
                                location_list_2_down[direction[1]] -= 2
                                location_2_down = get_location(integer_solution_num(PAM_2 - 2), location_list_2_down)
                                ang_dir_2 = ang_mom_2[direction[1]] - 1
                                RR[PAM_2][PAM_1][sequence_2][sequence_1] += (ang_dir_2 / (2 * q)) * \
                                                                            RR[PAM_2 - 2][PAM_1][location_2_down][
                                                                                sequence_1]
                                # fourth
                                location_list_1_up = ang_mom_1[:]  # search where i+1 in sequnce_1
                                location_list_1_up[direction[1]] += 1
                                location_1_up = get_location(integer_solution_num(PAM_1 + 1), location_list_1_up)
                                RR[PAM_2][PAM_1][sequence_2][sequence_1] -= (p / q) * \
                                                                            RR[PAM_2 - 1][PAM_1 + 1][direction[0]][
                                                                                location_1_up]
    return RR


# 重新读取之上字典，减少实际所需的键值
def trans_electron_transfer(eri_dict, a, b, c, d):
    """
    a function makes all necessary eri

    input:
    eri_dict： a dict about (i0|k0)
    a, b, c, d:the total angular moment of cartesian GTO

    output:
    a function makes all necessary eri
    """
    norm_eri_dict = {}
    for PAM_1 in range(a, a + b + 1):
        norm_eri_dict.setdefault(PAM_1, {})
        for PAM_2 in range(c, c + d + 1):
            norm_eri_dict[PAM_1].setdefault(PAM_2, {})
            for sequence_1, ang_mom_1 in enumerate(integer_solution_num(PAM_1)):
                norm_eri_dict[PAM_1][PAM_2].setdefault(sequence_1, {})
                for sequence_2, ang_mom_2 in enumerate(integer_solution_num(PAM_2)):
                    norm_eri_dict[PAM_1][PAM_2][sequence_1][sequence_2] = eri_dict[PAM_2][PAM_1][sequence_2][sequence_1]
    return norm_eri_dict


# form the （ij|k0）
# a dict about (ij|k0)
def HRR_1(eri_i0k0, Ra, Rb, a, b, c, d):
    """
    form the eri_ijk0

    Input:
    eri_i0k0:The dict is about eri_i0k0, i from a to a+b, j from c to c+d, but it is not a true electronic replusion
    integrals, just an auxiliary function.
    Ra: Coordinate of 1
    Rb: Coordinate of 2
    a, b, c, d:the total angular moment of cartesian GTO

    Output:
    form the eri_ijk0
    """
    '''
    a = tot_angular[orb_i]
    b = tot_angular[orb_j]
    c = tot_angular[orb_k]
    d = tot_angular[orb_l]
    '''
    eri = {}
    for PAM_3 in range(c, c + d + 1):
        eri.setdefault(PAM_3, {})
        for PAM_2 in range(0, 1):
            eri[PAM_3].setdefault(PAM_2, {})
            for PAM_1 in range(a, a + b + 1 - PAM_2):
                eri[PAM_3][PAM_2].setdefault(PAM_1, {})
                for sequence_3, ang_mom_3 in enumerate(integer_solution_num(PAM_3)):
                    eri[PAM_3][PAM_2][PAM_1].setdefault(sequence_3, {})
                    for sequence_2, ang_mom_2 in enumerate(integer_solution_num(PAM_2)):
                        eri[PAM_3][PAM_2][PAM_1][sequence_3].setdefault(sequence_2, {})
                        for sequence_1, ang_mom_1 in enumerate(integer_solution_num(PAM_1)):
                            eri[PAM_3][0][PAM_1][sequence_3][0][sequence_1] = eri_i0k0[PAM_1][PAM_3][sequence_1][
                                sequence_3]
    for PAM_3 in range(c, c + d + 1):
        eri.setdefault(PAM_3, {})
        for PAM_2 in range(1, b + 1):
            old_list = integer_solution_num(PAM_2 - 1)
            eri[PAM_3].setdefault(PAM_2, {})
            for PAM_1 in range(a, a + b + 1 - PAM_2):
                eri[PAM_3][PAM_2].setdefault(PAM_1, {})
                for sequence_3, ang_mom_3 in enumerate(integer_solution_num(PAM_3)):
                    eri[PAM_3][PAM_2][PAM_1].setdefault(sequence_3, {})
                    for sequence_2, ang_mom_2 in enumerate(integer_solution_num(PAM_2)):
                        eri[PAM_3][PAM_2][PAM_1][sequence_3].setdefault(sequence_2, {})
                        for sequence_1, ang_mom_1 in enumerate(integer_solution_num(PAM_1)):
                            direction = match_moment(old_list, ang_mom_2)
                            location_list_1_up = ang_mom_1[:]
                            location_list_1_up[direction[1]] += 1
                            location_1_up = get_location(integer_solution_num(PAM_1 + 1), location_list_1_up)
                            eri[PAM_3][PAM_2][PAM_1][sequence_3][sequence_2][sequence_1] = Ra[direction[1]] - Rb[
                                direction[1]]
                            eri[PAM_3][PAM_2][PAM_1][sequence_3][sequence_2][sequence_1] *= \
                                eri[PAM_3][PAM_2 - 1][PAM_1][sequence_3][direction[0]][sequence_1]
                            eri[PAM_3][PAM_2][PAM_1][sequence_3][sequence_2][sequence_1] += \
                                eri[PAM_3][PAM_2 - 1][PAM_1 + 1][sequence_3][direction[0]][location_1_up]
    eri_ab = {}
    for PAM_3 in range(c, c + d + 1):
        eri_ab.setdefault(PAM_3, {})
        for sequence_1, ang_mom_1 in enumerate(integer_solution_num(a)):
            eri_ab[PAM_3].setdefault(sequence_1, {})
            for sequence_2, ang_mom_2 in enumerate(integer_solution_num(b)):
                eri_ab[PAM_3][sequence_1].setdefault(sequence_2, {})
                for sequence_3, ang_mom_3 in enumerate(integer_solution_num(PAM_3)):
                    eri_ab[PAM_3][sequence_1][sequence_2][sequence_3] = eri[PAM_3][b][a][sequence_3][sequence_2][
                        sequence_1]
    return eri_ab


# a dict about (ij|k0)
def HRR_2(eri_ijk0, Rc, Rd, a, b, c, d):
    """
    a dict about (ij|kl)

    Input:
    eri_ijk0:The dict is about eri_ijk0, k from 0 to c+d
    integrals, just an auxiliary function.
    Rc: Coordinate of 3
    Rd: Coordinate of 4
    a, b, c, d:the total angular moment of cartesian GTO

    Output:
    a dict about (ij|k0)
    """
    eri = {}
    for sequence_1, ang_mom_1 in enumerate(integer_solution_num(a)):
        eri.setdefault(sequence_1, {})
        for sequence_2, ang_mom_2 in enumerate(integer_solution_num(b)):
            eri[sequence_1].setdefault(sequence_2, {})
            for PAM_4 in range(0, 1):
                eri[sequence_1][sequence_2].setdefault(PAM_4, {})
                for PAM_3 in range(c, c + d + 1 - PAM_4):
                    eri[sequence_1][sequence_2][PAM_4].setdefault(PAM_3, {})
                    for sequence_4, ang_mom_4 in enumerate(integer_solution_num(PAM_4)):
                        eri[sequence_1][sequence_2][PAM_4][PAM_3].setdefault(sequence_4, {})
                        for sequence_3, ang_mom_3 in enumerate(integer_solution_num(PAM_3)):
                            eri[sequence_1][sequence_2][0][PAM_3][0][sequence_3] = \
                                eri_ijk0[PAM_3][sequence_1][sequence_2][sequence_3]
    for sequence_1, ang_mom_1 in enumerate(integer_solution_num(a)):
        eri.setdefault(sequence_1, {})
        for sequence_2, ang_mom_2 in enumerate(integer_solution_num(b)):
            eri[sequence_1].setdefault(sequence_2, {})
            for PAM_4 in range(1, d + 1):
                eri[sequence_1][sequence_2].setdefault(PAM_4, {})
                old_list = integer_solution_num(PAM_4 - 1)
                for PAM_3 in range(c, c + d + 1 - PAM_4):
                    eri[sequence_1][sequence_2][PAM_4].setdefault(PAM_3, {})
                    for sequence_4, ang_mom_4 in enumerate(integer_solution_num(PAM_4)):
                        eri[sequence_1][sequence_2][PAM_4][PAM_3].setdefault(sequence_4, {})
                        for sequence_3, ang_mom_3 in enumerate(integer_solution_num(PAM_3)):
                            direction = match_moment(old_list, ang_mom_4)
                            location_list_3_up = ang_mom_3[:]
                            location_list_3_up[direction[1]] += 1
                            location_3_up = get_location(integer_solution_num(PAM_3 + 1), location_list_3_up)
                            eri[sequence_1][sequence_2][PAM_4][PAM_3][sequence_4][sequence_3] = Rc[direction[1]] - Rd[
                                direction[1]]
                            eri[sequence_1][sequence_2][PAM_4][PAM_3][sequence_4][sequence_3] *= \
                                eri[sequence_1][sequence_2][PAM_4 - 1][PAM_3][direction[0]][sequence_3]
                            eri[sequence_1][sequence_2][PAM_4][PAM_3][sequence_4][sequence_3] += \
                                eri[sequence_1][sequence_2][PAM_4 - 1][PAM_3 + 1][direction[0]][location_3_up]
    eri_abcd = {}
    for sequence_1, ang_mom_1 in enumerate(integer_solution_num(a)):
        eri_abcd.setdefault(sequence_1, {})
        for sequence_2, ang_mom_2 in enumerate(integer_solution_num(b)):
            eri_abcd[sequence_1].setdefault(sequence_2, {})
            for sequence_3, ang_mom_3 in enumerate(integer_solution_num(c)):
                eri_abcd[sequence_1][sequence_2].setdefault(sequence_3, {})
                for sequence_4, ang_mom_4 in enumerate(integer_solution_num(d)):
                    eri_abcd[sequence_1][sequence_2][sequence_3][sequence_4] = \
                        eri[sequence_1][sequence_2][d][c][sequence_4][sequence_3]
    return eri_abcd


# the function form the coefficience product
def trans_coeff_eri(info, orb_i, orb_j, orb_k, orb_l):
    """
    the function form the coefficience product

    Input:
    info: a dict about all kinds of information
    orb i, orb j, orb k, orb l:the orbitals of these atoms

    Output:
    a list about the product of four coefficiences
    """
    coeff_first = list(map(float, info['orbit_coeff'][orb_i]))
    coeff_second = list(map(float, info['orbit_coeff'][orb_j]))
    coeff_third = list(map(float, info['orbit_coeff'][orb_k]))
    coeff_fourth = list(map(float, info['orbit_coeff'][orb_l]))
    lier = [coeff_first[i] * coeff_second[j] * coeff_third[k] * coeff_fourth[l] for i in range(0, len(coeff_first))
            for j in range(0, len(coeff_second)) for k in range(0, len(coeff_third))
            for l in range(0, len(coeff_fourth))]
    return lier


# form a dict, whinch includes the value of different exponents group
def trans_int_eri(info, atom_i, atom_j, atom_k, atom_l, orb_i, orb_j, orb_k, orb_l, tot_angular):
    """
    form a dict, whinch includes the value of different exponents group

    Input:
    info: a dict about all kinds of information
    atom i, atom j, atom k, atom l: the sequences of these atoms
    orb i, orb j, orb k, orb l:the orbitals of these atoms
    tot_angular: a dict about the angular moment of the orbital

    Output:
    a dict, whinch includes the value of different exponents group
    """
    zeta_first = list(map(float, info[atom_i]['zeta'][orb_i]))
    zeta_second = list(map(float, info[atom_j]['zeta'][orb_j]))
    zeta_third = list(map(float, info[atom_k]['zeta'][orb_k]))
    zeta_fourth = list(map(float, info[atom_l]['zeta'][orb_l]))
    dict_counter = 0
    expon_dict = {}
    Ra = info[atom_i]['coordinate']
    Rb = info[atom_j]['coordinate']
    Rc = info[atom_k]['coordinate']
    Rd = info[atom_l]['coordinate']
    for alpha in zeta_first:
        for beta in zeta_second:
            for gamma in zeta_third:
                for delta in zeta_fourth:
                    eri_dict = ERI(Ra, Rb, Rc, Rd, orb_i, orb_j, orb_k, orb_l, tot_angular, alpha, beta, gamma, delta)
                    expon_dict.setdefault(dict_counter, eri_dict)
                    dict_counter += 1
    return expon_dict


# a dict about (ij|kl)
def ERI(Ra, Rb, Rc, Rd, orb_i, orb_j, orb_k, orb_l, tot_angular, alpha, beta, gamma, delta):
    """
    a dict about (ij|kl)

    Input:
    Ra: Coordinate of 1
    Rb: Coordinate of 2
    Rc: Coordinate of 3
    Rd: Coordinate of 4
    orb_i,orb_j,orb_k,orb_l:just like '2_P'
    tot_angular: a dict about the angular moment of orbitals
    alpha, beta, gamma, delta: Gaussian exponential value

    output:
    a dict about (ij|kl)
    """
    p, Rp, KAB = gaussian_product(alpha, beta, Ra, Rb)
    q, Rq, KCD = gaussian_product(gamma, delta, Rc, Rd)
    u = p * q / (p + q)
    a = tot_angular[orb_i]
    b = tot_angular[orb_j]
    c = tot_angular[orb_k]
    d = tot_angular[orb_l]
    tot_ang = a + b + c + d
    boy_array = boys_arr(tot_ang, u * np.dot(Rp - Rq, Rp - Rq))
    SSSS = SSSS_n(p, q, KAB, KCD, boy_array, tot_ang)
    eri_i000 = OS_i000(tot_ang, SSSS, Rp, Rq, Ra, u, p)
    eri_i0k0 = electron_transfer_RR(eri_i000, beta, delta, Ra, Rb, Rc, Rd, p, q, a, b, c, d)
    eri_i0k0 = trans_electron_transfer(eri_i0k0, a, b, c, d)
    eri_ijk0 = HRR_1(eri_i0k0, Ra, Rb, a, b, c, d)
    eri_ijkl = HRR_2(eri_ijk0, Rc, Rd, a, b, c, d)
    for sequence_1, ang_mom_1 in enumerate(integer_solution_num(a)):
        for sequence_2, ang_mom_2 in enumerate(integer_solution_num(b)):
            for sequence_3, ang_mom_3 in enumerate(integer_solution_num(c)):
                for sequence_4, ang_mom_4 in enumerate(integer_solution_num(d)):
                    normalization = norm(ang_mom_1[0], ang_mom_1[1], ang_mom_1[2], alpha)
                    normalization *= norm(ang_mom_2[0], ang_mom_2[1], ang_mom_2[2], beta)
                    normalization *= norm(ang_mom_3[0], ang_mom_3[1], ang_mom_3[2], gamma)
                    normalization *= norm(ang_mom_4[0], ang_mom_4[1], ang_mom_4[2], delta)
                    eri_ijkl[sequence_1][sequence_2][sequence_3][sequence_4] *= normalization
    return eri_ijkl


# a dict about eri of CGTO
def eri_orb_product(expon_dict, trans_coeff, orb_i, orb_j, orb_k, orb_l, tot_angular):
    """
    a dict about eri of CGTO

    input:
    expon_dict: a dict about (ij|kl)
    trans_coeff: 相应的系数乘积
    orb_i,orb_j,orb_k,orb_l:just like '2_P'
    tot_angular: a dict about the angular moment of orbitals

    output:
    a dict about eri of CGTO
    """
    eri = {}
    for sequence_1, ang_mom_1 in enumerate(integer_solution_num(tot_angular[orb_i])):
        eri.setdefault(sequence_1, {})
        for sequence_2, ang_mom_2 in enumerate(integer_solution_num(tot_angular[orb_j])):
            eri[sequence_1].setdefault(sequence_2, {})
            for sequence_3, ang_mom_3 in enumerate(integer_solution_num(tot_angular[orb_k])):
                eri[sequence_1][sequence_2].setdefault(sequence_3, {})
                for sequence_4, ang_mom_4 in enumerate(integer_solution_num(tot_angular[orb_l])):
                    eri[sequence_1][sequence_2][sequence_3][sequence_4] = 0
                    for i in range(len(trans_coeff)):
                        eri[sequence_1][sequence_2][sequence_3][sequence_4] += expon_dict[i][sequence_1][sequence_2][
                                                                                   sequence_3][sequence_4] * \
                                                                               trans_coeff[i]
    return eri


# 除去输入的(ij, kl)以外的集合对称数
def sym_number(i, j, k, l, ij, kl, counter):
    """
    形成一个对称数,用以减少实际计算量。

    input：
    i，j, k, l:四个整数
    ij = i*counter + j
    counter:the number of orbitals

    output:
    除去输入的(ij, kl)以外的集合对称数
    """
    num_pair = {(i * counter + j, k * counter + l),
                (i * counter + j, l * counter + k),
                (j * counter + i, k * counter + l),
                (j * counter + i, l * counter + k),
                (k * counter + l, i * counter + j),
                (k * counter + l, j * counter + i),
                (l * counter + k, i * counter + j),
                (l * counter + k, j * counter + i)}
    sub_pair = {(ij, kl)}
    num_pair = num_pair.__sub__(sub_pair)
    return num_pair


# ERI matrix
def ERI_mat(counter, ele_num, orb_order, tot_angular, info):
    """
    ERI matrix

    Input:
    counter:the number of total orbitals
    ele_num: the number of the atoms
    orb_order: the sequence of the orbital
    tot_angular: the angular moments of the orbitals
    info: a dict about all kinds of information

    Output:
    ERI matrix
    """
    eri_arr = np.zeros((counter * counter, counter * counter))
    orbital_counter_1 = 0
    cyc = 0
    for atom_i in range(1, ele_num + 1):
        for orbital_i in orb_order:
            if orbital_i in info[atom_i]['zeta'].keys():
                cyc += 1
                print(cyc)
                orbital_counter_2 = 0
                for atom_j in range(1, atom_i + 1):
                    if atom_j == atom_i:
                        threshold = get_location(orb_order, orbital_i)
                    else:
                        threshold = len(orb_order) - 1
                    for orbital_j in orb_order[0: threshold + 1]:
                        if orbital_j in info[atom_j]['zeta'].keys():
                            orbital_counter_3 = 0
                            for atom_k in range(1, ele_num + 1):
                                for orbital_k in orb_order:
                                    if orbital_k in info[atom_k]['zeta'].keys():
                                        if orbital_counter_1 < orbital_counter_3:
                                            break
                                        orbital_counter_4 = 0
                                        for atom_l in range(1, atom_k + 1):
                                            if atom_l == atom_k:
                                                threshold = get_location(orb_order, orbital_k)
                                            else:
                                                threshold = len(orb_order) - 1
                                            for orbital_l in orb_order[0: threshold + 1]:
                                                if orbital_l in info[atom_l]['zeta'].keys():
                                                    counter_12 = orbital_counter_1 * counter + orbital_counter_2
                                                    counter_34 = orbital_counter_3 * counter + orbital_counter_4
                                                    if counter_12 < counter_34:
                                                        break
                                                    expon_dict = trans_int_eri(info, atom_i, atom_j, atom_k, atom_l,
                                                                               orbital_i, orbital_j, orbital_k,
                                                                               orbital_l, tot_angular)
                                                    trans_coeff = trans_coeff_eri(info, orbital_i, orbital_j, orbital_k,
                                                                                  orbital_l)
                                                    eri = eri_orb_product(expon_dict, trans_coeff, orbital_i, orbital_j,
                                                                          orbital_k, orbital_l, tot_angular)
                                                    for counter_i in range(orbital_counter_1, orbital_counter_1 + len(
                                                            integer_solution(orbital_i, tot_angular))):
                                                        for counter_j in range(orbital_counter_2,
                                                                               orbital_counter_2 + len(
                                                                                   integer_solution(orbital_j,
                                                                                                    tot_angular))):
                                                            for counter_k in range(orbital_counter_3,
                                                                                   orbital_counter_3 + len(
                                                                                       integer_solution(orbital_k,
                                                                                                        tot_angular))):
                                                                for counter_l in range(orbital_counter_4,
                                                                                       orbital_counter_4 + len(
                                                                                           integer_solution(
                                                                                               orbital_l,
                                                                                               tot_angular))):
                                                                    counter_12 = counter_i * counter + counter_j
                                                                    counter_34 = counter_k * counter + counter_l
                                                                    ang_mom_i = counter_i - orbital_counter_1
                                                                    ang_mom_j = counter_j - orbital_counter_2
                                                                    ang_mom_k = counter_k - orbital_counter_3
                                                                    ang_mom_l = counter_l - orbital_counter_4
                                                                    eri_arr[counter_12][counter_34] = \
                                                                        eri[ang_mom_i][ang_mom_j][ang_mom_k][ang_mom_l]
                                                                    number_pair = sym_number(counter_i, counter_j,
                                                                                             counter_k, counter_l,
                                                                                             counter_12, counter_34,
                                                                                             counter)
                                                                    if len(number_pair) != 0:
                                                                        for counter12, counter34 in number_pair:
                                                                            eri_arr[counter12][counter34] = \
                                                                                eri_arr[counter_12][counter_34]
                                                    orbital_counter_4 += len(integer_solution(orbital_l, tot_angular))
                                                else:
                                                    break
                                        orbital_counter_3 += len(integer_solution(orbital_k, tot_angular))
                                    else:
                                        break
                            orbital_counter_2 += len(integer_solution(orbital_j, tot_angular))
                        else:
                            break
                orbital_counter_1 += len(integer_solution(orbital_i, tot_angular))
            else:
                break
    return eri_arr


"""
t1 = time.time()
alpha = 1.
beta = 1.
gamma = 1.
delta = 1.
Ra = [1., 0., 0.]
Rb = [0., 1., 0.]
Rc = [0., 1., 1.]
Rd = [0., 0., 1.]
tot_ang = 8
a = 2
b = 2
c = 2
d = 2
p, Rp, KAB = gaussian_product(alpha, beta, Ra, Rb)
q, Rq, KCD = gaussian_product(gamma, delta, Rc, Rd)
u = p * q / (p + q)
boy_array = boy_recurrence(tot_ang, u * np.dot(Rp - Rq, Rp - Rq))
SSSS = SSSS_n(p, q, KAB, KCD, boy_array, tot_ang)
OS = OS_i000(tot_ang, SSSS, Rp, Rq, Ra, u, p)
electron_transfer = electron_transfer_RR(OS, beta, delta, Ra, Rb, Rc, Rd, p, q, a, b, c, d)
lo_1 = get_location(integer_solution_num(2), [0, 1, 1])
lo_2 = get_location(integer_solution_num(2), [1, 0, 1])
TRT = trans_electron_transfer(electron_transfer, a, b, c, d)
# print(electron_transfer[3][2][lo_2][lo_1] * norm(0, 1, 1, alpha) * norm(0, 0, 0,
#                                                                        beta) * norm(0, 2, 1, gamma) * norm(0, 0, 0,
#                                                                                                            delta))
# print(TRT[2][3][lo_1][lo_2] * norm(0, 2, 1, alpha) * norm(0, 0, 0, beta) * norm(0, 1, 1, gamma) * norm(0, 0, 0, delta))
tot_angular = {'1_S': 0, '2_S': 0, '2_P': 1, '3_S': 0, '3_P': 1, '4_S': 0, '3_D': 2, '4_P': 1, '5_S': 0, '4_D': 2,
               '5_P': 1}
HRR = HRR_1(TRT, Ra, Rb, a, b, c, d)
eri_abcd = HRR_2(HRR, Rc, Rd, a, b, c, d)
print(HRR[2][0][0][0] * norm(2, 0, 0, alpha) * norm(2, 0, 0, beta) * norm(2, 0, 0, gamma) * norm(0, 0, 0, delta))
print(eri_abcd[lo_1][0][lo_2][0] * norm(0, 1, 1, alpha) * norm(2, 0, 0, beta) * norm(1, 0, 1, gamma) * norm(2, 0, 0, delta))
eri_ijkl = ERI(Ra, Rb, Rc, Rd, '3_D', '3_D', '3_D', '3_D', tot_angular, alpha, beta, gamma, delta)
print(eri_ijkl[lo_1][0][lo_2][0])
"""

"""
Ra = [1., 0., 0.]
Rb = [0., 1., 0.]
Rc = [0., 0., 1.]
alpha = 1.
beta = 1.
a = 2
b = 2
tot_ang = a + b
p, Rp, KAB = gaussian_product(alpha, beta, Ra, Rb)
boy_array = boys_arr(tot_ang, p * np.dot(Rp - Rc, Rp - Rc))
os_00 = nuclear_00(p, KAB, boy_array, tot_ang)
print(os_00[0]*norm(0, 0, 0, alpha) * norm(0, 0, 0, beta))
os_i0 = nuclear_i0(tot_ang, os_00, Rp, Rc, Ra, p)
print(os_i0[2][0] * norm(2, 0, 0, alpha) * norm(0, 0, 0, beta))
os_ij = nuclear_ij(os_i0, a, b, Ra, Rb)
print(os_ij[0][0] * norm(2, 0, 0, alpha) * norm(2, 0, 0, beta))
tot_angular = {'1_S': 0, '2_S': 0, '2_P': 1, '3_S': 0, '3_P': 1, '4_S': 0, '3_D': 2, '4_P': 1, '5_S': 0, '4_D': 2,
               '5_P': 1}
ei_ab = norm_nuclear(Ra, Rb, Rc, alpha, beta, '3_D', '3_D', tot_angular)
print(ei_ab[1][1])
"""
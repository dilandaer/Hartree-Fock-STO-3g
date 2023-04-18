import os
import numpy as np
import EI


# molden
def name_output_molden(file_path):
    """
    output:the name of ouput file
    """
    name = file_path.split('.')
    file_name = [name[0] + '_MO']
    file_name.append(file_name[0] + '.molden')
    return file_name


def atom_output(info, output_path, molden_name):
    """
    output:the atom information
    """
    print('[Molden Format]', file=output_path)
    print('[Title]', file=output_path)
    print(molden_name[0], file=output_path)
    print('{0:<8}{1}'.format('[Atoms]', 'AU'), file=output_path)
    fmt = '{:<8}{:<4}{:<4}{:^13}{:^15}{:>11}'
    for i in range(1, len(info.keys())):
        print(fmt.format(info[i]['element'], i, info[i]['atomic_number'],
                         float(info[i]['coordinate'][0]).__format__('3.8f'),
                         float(info[i]['coordinate'][1]).__format__('3.8f'),
                         float(info[i]['coordinate'][2]).__format__('3.8f')), file=output_path)
    return 0


def gto_output(info, output_path):
    """
    output:the GTO information
    """
    shell_dict = {'1_S': 's', '2_S': 's', '2_P': 'p', '3_S': 's', '3_P': 'p', '4_S': 's', '3_D': 'd', '4_P': 'p',
                  '5_S': 's', '4_D': 'd', '5_P': 'p'}
    print('[GTO]', file=output_path)
    for i in range(1, len(info.keys())):
        print('{:>6}{:>6}'.format(i, '0'), file=output_path)
        for j in info[i]['zeta'].keys():
            print('{:<4}{:<2}{}'.format(shell_dict[j], len(info[i]['zeta'][j]), '1.0'), file=output_path)
            for k in range(len(info[i]['zeta'][j])):
                print('{:^18}{}'.format(info[i]['zeta'][j][k].__format__('.8E'),
                                        info['orbit_coeff'][j][k].__format__('.8E')), file=output_path)
        print(' ', file=output_path)
    return 0


def mo_output(scf_info, output_path, counter, number_electrons):
    """
    output:the orbtial energy and orbtal coefficients
    """
    print('\n[MO]', file=output_path)
    for i in range(counter):
        print('Ene={:>16}'.format(scf_info['orb_energy'][i].__format__('7.8f')), file=output_path)
        print('Spin= Alpha', file=output_path)
        if i < int(number_electrons / 2):
            print('Occup=  2.000000', file=output_path)
        else:
            print('Occup=  0.000000', file=output_path)
        for j in range(counter):
            print('{:>6}{:>18}'.format(j + 1, scf_info['orb_coeff'][j][i].__format__('7.10f')), file=output_path)
    return 0


def molden_ot(file_path, info, scf_info, counter, number_electrons):
    """
    put program together,输出molden文件

    input:
    file_path:输入文件的路劲
    info:a dict about all kinds of information
    scf_info:自洽场程序输出的字典
    counter:轨道的数量
    number_electrons:电子数量

    output:
    molden文件
    """
    molden_name = name_output_molden(file_path)
    molden_output = open(molden_name[1], 'w')
    atom_output(info, molden_output, molden_name)
    gto_output(info, molden_output)
    mo_output(scf_info, molden_output, counter, number_electrons)
    molden_output.close()
    return 0


def name_output_EI(file_path):
    """
    output:the name of ouput file
    """
    name = file_path.split('.')
    file_name = [name[0] + '_EI']
    file_name.append(file_name[0] + '.EI')
    return file_name


def eletron_integral_out(file_path, info, counter, atom_num, overlap_matrix, kinetic_matrix, nuclear_matrix, eri_matrix,
                         Hcore_mat, keywords):
    """
    输出EI文件

    input:
    file_path:输入文件的路劲
    info:a dict about all kinds of information
    counter:轨道的数量
    atom_num:原子数量
    overlap_matrix:S:重叠矩阵
    kinetic_matrix:动能矩阵
    nuclear_matrix:核势能矩阵的字典
    eri_matrix:电子排斥积分矩阵
    Hcore_mat:核心Hamilton矩阵
    keyword:输入文件中的一些关键词

    output:
    输出EI文件
    """
    criteria = 'nonEI'
    for ele in keywords:
        if ele == 'EI'or ele == 'ei':
            criteria = 'EI'
    if criteria == 'EI':
        EI_name = name_output_EI(file_path)
        EI_output = open(EI_name[1], 'w')
        orb_order = ['1_S', '2_S', '2_P', '3_S', '3_P', '4_S', '3_D', '4_P', '5_S', '4_D', '5_P']
        tot_angular = {'1_S': 0, '2_S': 0, '2_P': 1, '3_S': 0, '3_P': 1, '4_S': 0, '3_D': 2, '4_P': 1, '5_S': 0,
                       '4_D': 2,
                       '5_P': 1}
        print('overlap', file=EI_output)
        for i in range(counter):
            print(i, overlap_matrix[i], file=EI_output)
        print('kinetic', file=EI_output)
        for i in range(counter):
            print(i, kinetic_matrix[i], file=EI_output)
        H = np.zeros((counter, counter))
        for i in range(1, atom_num + 1):
            H += nuclear_matrix[i]
        print('nuclear', file=EI_output)
        for i in range(counter):
            print(i, H[i], file=EI_output)
        print('eri', file=EI_output)
        orbital_counter_1 = -1
        for atom_i in range(1, atom_num + 1):
            for orbital_i in orb_order:
                if orbital_i in info[atom_i]['zeta'].keys():
                    for angular_moment_i in EI.integer_solution(orbital_i, tot_angular):
                        orbital_counter_1 += 1
                        orbital_counter_2 = 0
                        for atom_j in range(1, atom_i + 1):
                            if atom_j == atom_i:
                                threshold = EI.get_location(orb_order, orbital_i)
                            else:
                                threshold = len(orb_order) - 1
                            for orbital_j in orb_order[0: threshold + 1]:
                                if atom_j == atom_i and orbital_j == orbital_i:
                                    threshold_orb = EI.get_location(EI.integer_solution(orbital_j, tot_angular),
                                                                    angular_moment_i)
                                else:
                                    threshold_orb = len(EI.integer_solution(orbital_j, tot_angular)) - 1
                                if orbital_j in info[atom_j]['zeta'].keys():
                                    for angular_moment_j in EI.integer_solution(orbital_j, tot_angular)[
                                                            0: threshold_orb + 1]:
                                        orbital_counter_3 = -1
                                        for atom_k in range(1, atom_num + 1):
                                            for orbital_k in orb_order:
                                                if orbital_k in info[atom_k]['zeta'].keys():
                                                    for angular_moment_k in EI.integer_solution(orbital_k, tot_angular):
                                                        orbital_counter_3 += 1
                                                        if orbital_counter_1 < orbital_counter_3:
                                                            break
                                                        orbital_counter_4 = 0
                                                        for atom_l in range(1, atom_k + 1):
                                                            if atom_l == atom_k:
                                                                threshold = EI.get_location(orb_order, orbital_k)
                                                            else:
                                                                threshold = len(orb_order) - 1
                                                            for orbital_l in orb_order[0: threshold + 1]:
                                                                if atom_l == atom_k and orbital_l == orbital_k:
                                                                    threshold_orb = EI.get_location(
                                                                        EI.integer_solution(orbital_l, tot_angular),
                                                                        angular_moment_k)
                                                                else:
                                                                    threshold_orb = len(
                                                                        EI.integer_solution(orbital_l, tot_angular)) - 1
                                                                if orbital_l in info[atom_l]['zeta'].keys():
                                                                    for angular_moment_l in EI.integer_solution(
                                                                            orbital_l,
                                                                            tot_angular)[
                                                                                            0: threshold_orb + 1]:
                                                                        counter_12 = orbital_counter_1 * counter + orbital_counter_2
                                                                        counter_34 = orbital_counter_3 * counter + orbital_counter_4
                                                                        if counter_12 < counter_34:
                                                                            break
                                                                        print(orbital_counter_1 + 1,
                                                                              orbital_counter_2 + 1,
                                                                              orbital_counter_3 + 1,
                                                                              orbital_counter_4 + 1,
                                                                              eri_matrix[counter_12][counter_34],
                                                                              file=EI_output)
                                                                        orbital_counter_4 += 1
                                                                else:
                                                                    break
                                                else:
                                                    break
                                        orbital_counter_2 += 1

                                else:
                                    break
                else:
                    break
        print('Hcore', file=EI_output)
        for i in range(counter):
            print(i, Hcore_mat[i], file=EI_output)

import numpy as np
import read
import sys
import time
import SCF_0
import output
import EI


np.set_printoptions(threshold=np.inf)
t1 = time.time()
# output: some information from input file and put the zeta of STO-3G into dict
file_path = sys.argv[1]
read_input = read.read(file_path)
read_input.read_all()
info = read_input.info
# so far, the content of dict is {1：{element:H, coordinate:[... , ... , ...], zeta: {'1_S': [854.0324951,
# 155.5630851, 42.10144179], '2_S':..., }, atomic_number:**}, 2: {}....... , 'orbit_coeff': '1_S': [0.1543289673,
# 0.5353281423, 0.4446345422], '2_S': [-0.09996722919, 0.3995128261, 0.7001154689]....}
atom_num = read_input.atom_num
print(atom_num)
number_electrons = read_input.electrons
print(number_electrons)
counter = read_input.counter
print(counter)
nuclear_charge = read_input.nucl_charge_dict
print(nuclear_charge)
keywords = read_input.keywords
print('{:15} | {:15} | {:15} | {:15}'.format('elements', 'coordinate_x', 'coordinate_y', 'coordinate_z'))
fmt = '{:15} | {:15} | {:15} | {:15}'
for i in range(1, atom_num + 1):
    print(fmt.format(info[i]['element'], float(info[i]['coordinate'][0]).__format__('3.8f'),
                     float(info[i]['coordinate'][1]).__format__('3.8f'),
                     float(info[i]['coordinate'][2]).__format__('3.8f')))

orb_order = ['1_S', '2_S', '2_P', '3_S', '3_P', '4_S', '3_D', '4_P', '5_S', '4_D', '5_P']
tot_angular = {'1_S': 0, '2_S': 0, '2_P': 1, '3_S': 0, '3_P': 1, '4_S': 0, '3_D': 2, '4_P': 1, '5_S': 0, '4_D': 2,
               '5_P': 1}
# Overlap
# for any atom
# 对于每个原子进行循环

overlap_matrix = EI.overlap_mat(counter, atom_num, orb_order, tot_angular, info)
kinetic_matrix = EI.kinetic_mat(counter, atom_num, orb_order, tot_angular, info)
nuclear_matrix = EI.nuclear_mat(counter, atom_num, orb_order, tot_angular, info)
eri_matrix = EI.ERI_mat(counter, atom_num, orb_order, tot_angular, info)
Hcore_mat = SCF_0.Hcore(kinetic_matrix, nuclear_matrix, atom_num)
scf_info = SCF_0.SCF(Hcore_mat, eri_matrix, overlap_matrix, number_electrons, info, nuclear_charge)
output.molden_ot(file_path, info, scf_info, counter, number_electrons)
output.eletron_integral_out(file_path, info, counter, atom_num, overlap_matrix, kinetic_matrix, nuclear_matrix, eri_matrix,
                         Hcore_mat, keywords)
print(scf_info['total_energy'])
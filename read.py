import re
import Basis


class read:
    def __init__(self, file_path):
        self.file_path = file_path
        self.info = {}
        self.atom_num = 0
        self.electrons = 0
        self.nucl_charge_dict = {}
        self.counter = 0
        self.keywords = []

    def position(self):
        """
        get the position of atoms
        """
        with open(self.file_path, 'r') as fp:
            contents = fp.read()
        re_content = re.finditer(r'([A-Z][a-z]?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)', contents)
        for seq_num, content in enumerate(re_content, 1):
            self.atom_num += 1
            self.info.setdefault(seq_num, {})
            self.info[seq_num].setdefault('element', content.group(1))
            coordinate = list(map(float, [content.group(2), content.group(3), content.group(4)]))
            coordinate = [coordinate[i] / 0.5291772083 for i in range(3)]
            self.info[seq_num].setdefault('coordinate', coordinate)

    def nuclear_charge(self):
        """
        define the atomic number
        """
        file_path = 'sequence.txt'
        with open(file_path, 'r') as fp:
            contents = fp.read()
        re_content = re.findall(r'([A-Z][a-z]?)', contents)
        for nucl_charge, element in enumerate(re_content, 1):
            self.nucl_charge_dict.setdefault(element, nucl_charge)

    def get_electrons(self):
        """
        get electrons from charge
        """
        with open(self.file_path, 'r') as fp:
            contents = fp.read()
        re_content = re.findall(r'(-?\d+)\s+(\d+)', contents)
        re_content = list(map(int, re_content[0]))
        for i in range(1, self.atom_num+1):
            self.electrons += self.nucl_charge_dict[self.info[i]['element']]
        self.electrons -= re_content[0]

    def read_basis(self):
        """
        get the data of basis
        """
        for i in range(1, self.atom_num + 1):
            self.info[i].setdefault('zeta', Basis.read_basis()['basis_zeta'][self.info[i]['element']])
            self.info[i].setdefault('atomic_number', self.nucl_charge_dict[self.info[i]['element']])
        self.info.setdefault('orbit_coeff', Basis.read_basis()['orbit_coeff'])

    def counter_number(self):
        """
        counter: the number of orbitals
        """
        number = 0
        orb_order = ['1_S', '2_S', '2_P', '3_S', '3_P', '4_S', '3_D', '4_P', '5_S', '4_D', '5_P']
        tot_angular = {'1_S': 1, '2_S': 1, '2_P': 3, '3_S': 1, '3_P': 3, '4_S': 1, '3_D': 6, '4_P': 3, '5_S': 1,
                       '4_D': 6, '5_P': 3}
        for i in range(1, self.atom_num + 1):
            for orb_name_i in orb_order:
                if orb_name_i in self.info[i]['zeta'].keys():
                    number += tot_angular[orb_name_i]
                else:
                    break
        self.counter = number

    def get_keywords(self):
        """
        get keywords
        """
        with open(self.file_path, 'r') as fp:
            contents = fp.read()
        re_content = re.findall(r'key:\s+(\w+)\s+(\w+)\s+(\w+)\s+end', contents)
        for ele in re_content[0]:
            self.keywords.append(ele)

    def read_all(self):
        """
        read all
        """
        self.position()
        self.nuclear_charge()
        self.get_electrons()
        self.read_basis()
        self.counter_number()
        self.get_keywords()

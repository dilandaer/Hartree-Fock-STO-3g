import re


# 读取基组数据库文件中的元素Zeta值和轨道系数
def read_basis(file_path='sto-3g.txt'):
    """
    函数参数：
    file_path    基组数据库文件的路径，默认与该py文件位于同一目录下且文件名为"sto-3g.txt"，str

    有返回值：
    re_basis     读取基组数据库文件的结果，re_basis['basis_zeta']为元素Zeta值字典，re_basis['orbit_coeff']为轨道系数字典，dict
    """

    # 打开数据库文件读取内容并删除所有换行符与空格
    with open(file_path, 'r') as fp:
        content = fp.read().replace('\n', '').replace(' ', '')

    # 使用正则表达式进行匹配，结果模式示例: ('H', 'S', '0.3425250914E+01...0.4446345422E+00')，re_content为包含所有匹配结果的列表
    re_content = re.findall(r'([A-Z][a-z]?)([SD]P?)(0[\dE\+-.]+)', content)

    # 读取数据库中的所有元素，初始化元素Zeta值字典和轨道系数字典
    elements = list(set([rc[0] for rc in re_content]))
    basis_zeta = {ele: {} for ele in elements}
    orbit_coeff = {}

    # 对所有元素进行遍历
    for ele in elements:

        # 从re_content列表中读取指定元素的所有匹配结果，初始化壳层编号i(SP)、j(D)
        ele_basis = [rc for rc in re_content if rc[0] == ele]
        i = 2
        j = 3

        # 对指定元素的匹配结果进行读取并写入元素Zeta值字典basis_zeta
        for bs in ele_basis:

            # 对一次正则匹配的第三项数字部分进行二次匹配分割并转换为浮点数
            sci_number = re.findall(r'(-?0.\d+E[\+-]\d{2})', bs[2])
            float_number = [float(num) for num in sci_number]

            # 根据轨道类型不同选择不同的读取方式和编号方式并写入basis_zeta
            if bs[1] == 'S':
                zeta_list = [float_number[i] for i in [0, 2, 4]]
                basis_zeta[ele].update({'1_S': zeta_list})

            elif bs[1] == 'SP':
                zeta_list = [float_number[i] for i in [0, 3, 6]]
                basis_zeta[ele].update({f'{i}_S': zeta_list})
                basis_zeta[ele].update({f'{i}_P': zeta_list})
                i += 1

            elif bs[1] == 'D':
                zeta_list = [float_number[i] for i in [0, 2, 4]]
                basis_zeta[ele].update({f'{j}_D': zeta_list})
                j += 1

    # 读取元素Xe的各轨道系数并写入轨道系数字典orbit_coeff，读取写入模式与上述方法一致
    xe_basis = [rc for rc in re_content if rc[0] == 'Xe']
    i = 2
    j = 3

    for xe in xe_basis:

        sci_number = re.findall(r'(-?0.\d+E[\+-]\d{2})', xe[2])
        float_number = [float(num) for num in sci_number]

        if xe[1] == 'S':
            coeff_list = [float_number[i] for i in [1, 3, 5]]
            orbit_coeff['1_S'] = coeff_list

        elif xe[1] == 'SP':
            coeff_list = [float_number[i] for i in [1, 4, 7]]
            orbit_coeff[f'{i}_S'] = coeff_list
            coeff_list = [float_number[i] for i in [2, 5, 8]]
            orbit_coeff[f'{i}_P'] = coeff_list
            i += 1

        elif xe[1] == 'D':
            coeff_list = [float_number[i] for i in [1, 3, 5]]
            orbit_coeff[f'{j}_D'] = coeff_list
            j += 1

    re_basis = {'basis_zeta': basis_zeta, 'orbit_coeff': orbit_coeff}

    return re_basis


# 查看读取效果
if __name__ == '__main__':
    print(read_basis()['basis_zeta']['H'])
    print(read_basis()['orbit_coeff']['1_S'])

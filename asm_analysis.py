import os
import csv
import sys
import pandas as pd


def get_function_list(file_input):
    func_str_list = []
    with open(file_input, 'r') as file:
        input_string = file.read()
        lines = input_string.split('\n')
        for line in lines:
            if "__device__" in line:
                func_str_list.append(line)
    return func_str_list


def create_test_kernel(function_str):
    data_type = ['int', 'unsigned', 'double', 'float']
    hack_input = ['x', 'y', 'z', 'w']
    hack_input_index = 0
    function_issue = False

    cuda_header = '''#include <math.h>\n'''
    str_split = function_str.split("(")
    # input parameters, should be used in function param
    input_param = str_split[-1][:-1]
    # input element, should be used in function call
    input_element = []
    for item in input_param.split(','):
        if item.split(' ')[-1] != '':
            input_element.append(item.split(' ')[-1])
        else:
            input_element.append(item.split(' ')[-2])

    # sometime nvidia cuda spec has some typo issue, like '__device__? double fmax ( double , double )'
    for item in range(len(input_element)):
        if input_element[item] in data_type:
            function_issue = True
            input_element[item] = hack_input[item]
    if function_issue:
        parts = input_param.split(',')
        input_param = ', '.join([part.strip() + ' ' + param for part, param in zip(parts, hack_input)])

    return_type = ' '.join(str_split[0].split(" ")[1:-2])
    if return_type == '__RETURN_TYPE':
        return_type = 'int'
    elif return_type == 'void':
        return_type = ''
    function_name = str_split[0].split(" ")[-2].lstrip().rstrip()

    if return_type:
        function_header = '__global__ void ' + 'test_' + function_name \
                          + '(' + input_param + ', ' + return_type + ' *output' + ' ) {'
        function_body = '*output = ' + function_name + '('  + ', '.join(input_element) + ');'
        function_tail = '}\n'
    else:
        function_header = '__global__ void ' + 'test_' + function_name \
                          + '(' + input_param + ' ) {'
        function_body = function_name + '(' + ', '.join(input_element) + ');'
        function_tail = '}\n'

    main_func = 'int main(){return 0;}'

    function_call = cuda_header + function_header + function_body + function_tail + main_func
    return function_call, function_name, return_type


def create_test_cu_file(function_call, file_path):
    file_content = function_call
    with open(file_path, 'w') as file:
        file.write(file_content)


def exec_command(plantform, file_path):
    file_output = file_path.replace('.cu', '_' + plantform + '.o')
    if plantform == 'nv':
        # compile_command = 'nvcc -arch=sm_86 -rdc=true -Xcompiler -fPIC --shared ' + file_path + ' -o ' + file_output
        compile_command = 'nvcc -arch=sm_86  ' + file_path + ' -o ' + file_output
        dump_command = 'cuobjdump -sass ' + file_output + ' > test_nv.sass'
    elif plantform == 'dl':
        compile_command = ''
        dump_command = ''
    else:
        assert 0
    return os.system(compile_command) + os.system(dump_command)


def cal_instruction_number(plantform, file_path):
    if plantform == 'nv':
        file_input = file_path.replace('.cu', '_nv.sass')
        with open(file_input, 'r') as file:
            input_string = file.read()
            lines = input_string.split('\n')
            count = 0
            in_comment_block = False

            for line in lines:
                if 'test' in line:
                    count = 0
                if line.strip().startswith("/*") and line.strip().endswith("*/"):
                    if 'NOP' in line or 'EXIT' in line:
                        count += 1
                        break
                    count += 1
                elif line.strip().startswith("/*"):
                    in_comment_block = True
                elif line.strip().endswith("*/"):
                    count += 1
                    in_comment_block = False
                elif in_comment_block:
                    count += 1
        return count / 2
    elif plantform == 'dl':
        file_input = file_path.replace('.cu', '_dl.sass')
        with open(file_input, 'r') as file:
            input_string = file.read()
            lines = input_string.split('\n')
            count = 0

            for line in lines:
                if 'test' in line:
                    count = 0
                elif 'kill_a' in line:
                    count += 1
                    break
                else:
                    count += 1
        return count


def delete_files(plantform, file_path):
    file_input = file_path
    file_output = file_path.replace('.cu', '_' + plantform + '.o')
    file_dump = file_path.replace('.cu', '_' + plantform + '.sass')
    try:
        os.remove(file_input)
        os.remove(file_output)
        os.remove(file_dump)
    except OSError as e:
        print('delete failed :', e)


if __name__ == '__main__':
    args = sys.argv[1:]
    plantform = args[0]

    file_list = []
    try:
        file_name = args[1]
        print(file_name)
    except IndexError:
        # no file specified, walk through the folder
        for file in os.listdir(os.getcwd()):
            if '.api' in file:
                file_list.append(file)
    else:
        file_list.append(file_name)

    for file in file_list:
        file_dump_result = pd.DataFrame()

        # open api file
        function_file_path = os.getcwd() + '/' + file
        # get function list
        function_list = get_function_list(function_file_path)

        for function_str in function_list:
            # create kernel code
            kernel_code, function_name, function_return_type = create_test_kernel(function_str.replace('?', ''))
            # write to cuda file
            file_path = os.getcwd() + '/test.cu'
            create_test_cu_file(kernel_code, file_path)
            # compile cuda file
            failed = exec_command(plantform, file_path)
            if not failed:
                # calculate instructions
                inst_number = cal_instruction_number(plantform, file_path)
                # remove generated files
                delete_files(plantform, file_path)
                dump_result = {'function_name': function_name, 'function_return_type': function_return_type,
                               'inst_number': inst_number}
            else:
                dump_result = {'function_name': function_name, 'function_return_type': 'NOT SUPPORTED',
                               'inst_number': 0}
            file_dump_result = file_dump_result.append(dump_result, ignore_index=True)
        file_dump_result['inst_number'] = file_dump_result['inst_number'].astype(int)
        output_file_name = plantform + '_' + file.split('.')[0] + '.csv'
        output_file_path = os.getcwd() + '/' + output_file_name
        file_dump_result.to_csv(output_file_path, index=True, quoting=csv.QUOTE_NONNUMERIC)


import numpy as np
from re import match


def csv_to_array(s):
    lines = [match('(^[^\n]+)\n?$', line).group(1) for line in s.split('\n')]

    separator = ','
    data = np.zeros(shape = (len(lines), len(lines[0].split(separator))))
    for i in range(0, len(lines)):
        values = [float(value) for value in lines[i].split(separator)]
        data[i,:] = values

    return data

def array_to_csv(a):
    lines = []
    separator = ','
    for i in range(0, a.shape[0]):
        lines.append(separator.join([str(value) for value in a[i,:]]))
    return '\n'.join(lines)
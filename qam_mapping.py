import numpy as np

# QAM mapping
re = [-3, -1, +3, +1]
im = [-3, -1, +3, +1]
mapping_table = {}
demapping_table = {}
i = 0
for r in re:
    for im_ in im:
        bits = [int(x) for x in np.binary_repr(i, width=4)]
        symbol = (r + 1j * im_) / np.sqrt(10)
        mapping_table[tuple(bits)] = symbol
        demapping_table[symbol] = bits
        i += 1

def Demapping(QAM):
    constellation = np.array(list(demapping_table.keys()))
    dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1)))
    const_index = dists.argmin(axis=1)
    hardDecision = constellation[const_index]
    return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision

def PS(bits):
    return bits.reshape((-1,))
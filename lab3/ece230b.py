import random
import math
import numpy as np

def gen_rand_qam_symbols(N, M):
    '''
    Inputs:
    N: Number of random symbols to generate
    M: Order of the QAM constellation (e.g., 4 for QPSK, 16 for 16-QAM)
    
    Outputs:
    symbols: N randomly selected M-QAM symbols.
    constellation: The full M-QAM constellation.
    '''

    # this works but there's no need for the gray coding
    '''
    for i in range(N):
        s = random.randint(0, M - 1)
        if M == 16:
            symbols.append('{0:04b}'.format(s))
        elif M == 4:
            symbols.append('{0:02b}'.format(s))
    '''
        
    bits = int(math.sqrt(M))
    symbols = []
    constellation = []

    # generate constellation
    for a in range(bits):
        a = (2*a) + 1 - bits
        for b in range(bits):
            b = (2*b) + 1 - bits
            # normalize with unit average energy
            A = math.sqrt(3/(2*(M-1)))
            constellation.append(complex(A*a, A*b))
    
    for i in range(N):
        s = random.randint(0, M - 1)
        symbols.append(constellation[s])

    return (symbols, constellation)

def create_pulse_train(symbols, sps):
    '''
    Inputs:
    symbols: A sequence of symbols.
    sps: Samples per symbol, i.e., the number of discrete-time samples from one symbol to the next.
    
    Output:
    pulse_train: A discrete-time signal where each symbol is separated by (sps-1) zeros.
    '''
    zero_arr = [complex(0, 0)]*(len(symbols) * (sps - 1)) # one less than sps because we have to insert nonzero numbers
    indices = [(i * sps) for i in range(len(symbols))]
    counter = 0
    for i in indices:
        zero_arr.insert(i, symbols[counter])
        counter += 1
    return zero_arr

def get_rc_pulse(beta, span, sps):
    '''
    Inputs:
    beta: Rolloff factor 𝛽 (between 0 and 1, inclusive).\n
    span: The integer number of symbols spanned by the pulse, not including the symbol at 𝜏=0.\n
    sps: Samples per symbol, as defined previously.

    Output:
    pulse: A truncated raised cosine pulse, symmetric and centered at 𝜏=0. The number of zero crossings should be equal to span. The pulse should be normalized such that its energy is equal to one.
    '''
    n = np.arange(-span * sps, span * sps + 1) # arange is end exlusive
    t = n / sps

    with np.errstate(divide='ignore', invalid='ignore'):
        numerator   = np.sinc(t) * np.cos(beta * np.pi * t)
        denominator = 1 - (2 * beta * t) ** 2
        h = numerator / denominator
    
    if beta > 0:
        t_special = 1 / (2 * beta)
        special_mask = np.isclose(np.abs(t), t_special)
        h[special_mask] = (np.pi / 4) * np.sinc(1 / (2 * beta))
    
    # Normalize to unit energy
    energy = np.sum(h**2) / sps
    h = h / np.sqrt(energy)

    return h

def get_rrc_pulse(beta, span, sps):
    '''
    Inputs:
    beta: Rolloff factor 𝛽 (between 0 and 1, inclusive).
    span: The integer number of symbols spanned by the pulse, not including the symbol at 𝜏=0.
    sps: Samples per symbol

    Output:
    A truncated root raised cosine pulse, symmetric and centered at 𝜏=0. The number of zero crossings should be equal to span. The pulse should be normalized such that its energy is equal to one.\n
    Note that the period is 10 microseconds
    '''
    n = np.arange(-span * sps, span * sps + 1) # arange is end exlusive
    T = 1/(10**5) # symbol period
    t = n * T / sps

    with np.errstate(divide='ignore', invalid='ignore'):
        numerator   = np.sin(np.pi * (t/T) * (1 - beta)) + ((4 * beta * (t/T)) * np.cos(np.pi * (t/T) * (1 + beta)))
        denominator = np.pi * (t/T) * (1 - ((4 * beta * (t/T)) ** 2))
        h = (1/np.sqrt(T)) * (numerator / denominator)

    if beta > 0:
        t_special = T / (4 * beta)
        special_mask = np.isclose(np.abs(t), t_special)
        h[special_mask] = (beta/np.sqrt(T * 2)) * (((1 + (2/np.pi)) * np.sin(np.pi/(4 * beta))) + ((1 - (2/np.pi)) * np.cos(np.pi/(4 * beta))))

    # rrc also has a special value at 0, which in our case is the middle.
    h[len(n)//2] = (1/np.sqrt(T)) * (1 - beta + (4 *(beta / np.pi)))

    # normalize to unit energy
    energy = np.sum(h**2) / sps
    h = h / np.sqrt(energy)

    return h

def symbol_error_rate(received_symbols, transmitted_symbols, constellation):
    """
    Inputs:
    received_symbols : the received and equalized symbols
    transmitted_symbols : the original transmitted symbols
    constellation : square constellation

    Output:
    ser: float, the symbol error rate
    """
    received_symbols    = received_symbols
    transmitted_symbols = transmitted_symbols
    constellation       = constellation

    # For each received symbol, find the nearest constellation point
    distances = np.abs(received_symbols[:, np.newaxis] - constellation[np.newaxis, :])
    decided_indices = np.argmin(distances, axis=1)
    decided_symbols = constellation[decided_indices]

    errors = np.sum(decided_symbols != transmitted_symbols)
    ser = errors / len(transmitted_symbols)
    return ser
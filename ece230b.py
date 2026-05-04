import random
import math

def gen_rand_qam_symbols(N, M):
    '''
    Inputs:
    N: Number of random symbols to generate
    M: Order of the QAM constellation (e.g., 4 for QPSK, 16 for 16-QAM)
    
    Outputs:
    symbols: N randomly selected M-QAM symbols.
    constellation: The full M-QAM constellation.
    '''

    bits = math.sqrt(M)
    symbols = []
    constellation = []
    for i in range(N):
        s = random.randint(0, M - 1)
        if M == 16:
            symbols.append('{0:04b}'.format(s))
        elif M == 4:
            symbols.append('{0:02b}'.format(s))
        
    # generate constellation
    for a in range(bits):
        for b in range(bits):
            a = (2*a) + 1 - bits
            b = (2*b) + 1 - bits
            # normalize with unit average energy
            A = math.sqrt(3/(2*(M-1)))
            constellation.append(complex(A*a, A*b))

    return (symbols, constellation)

def create_pulse_train(symbols, sps):
    '''
    Inputs:
    symbols: A sequence of symbols.
    sps: Samples per symbol, i.e., the number of discrete-time samples from one symbol to the next.
    
    Output:
    pulse_train: A discrete-time signal where each symbol is separated by (sps-1) zeros.
    '''
    zero_arr = [str(0)]*(len(symbols) * (sps - 1)) # one less than sps because we have to insert nonzero numbers
    indices = [(i * sps) for i in range(len(symbols))]
    counter = 0
    for i in indices:
        zero_arr.insert(i, symbols[counter])
        counter += 1
    return zero_arr

def get_rc_pulse(beta, span, sps):
    '''
    Inputs:
    beta: Rolloff factor 𝛽 (between 0 and 1, inclusive).
    span: The integer number of symbols spanned by the pulse, not including the symbol at 𝜏=0.
    sps: Samples per symbol, as defined previously.

    Output:
    pulse: A truncated raised cosine pulse, symmetric and centered at 𝜏=0. The number of zero crossings should be equal to span. The pulse should be normalized such that its energy is equal to one.
    '''
    pass

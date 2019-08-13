"""ber.py
"""
import numpy as np
from scipy.special import erfc


def bpsk_ber(ebno_db):
    """
    Return theoretical BER at EbNo_dB

    Args:
         ebno_db : Vector of Eb/No values in dB
    """
    ebno = 10 ** (ebno_db / 10.0)
    return 0.5 * erfc(np.sqrt(ebno))


def fm0_ber(ebno_db):
    """
    Return theoretical BER for FM0 (Bi-phase space) using symbol-by-symbol
    decoding

    Error probability taken from eq(8) of Marvin Simon, Dariush Divsalar,
    "Some Interesting Observations for Certain Line Codes With Application to
    RFID", IEEE Transactions on Communications, Vol 54., No. 4, April 2006

    """
    ebno = 10 ** (ebno_db / 10.0)
    sqrt_ebno_2 = np.sqrt(ebno/2.0)
    return erfc(sqrt_ebno_2) * (1.0 - 0.5 * erfc(sqrt_ebno_2))


def get_noise_std_deviation(eb, ebno_db_target):
    """Return noise standard deviation given Eb and target Eb/No

    var(noise) = sigma**2 = No/2 per dimension
    stddev(noise) = sqrt(var(noise)) = sigma

    Args:
        eb : Energy per bit (linear)
        ebno_db_target : Eb/No target value in dB

    Returns:
        sqrt(no/2) : Noise standard deviation
    """
    ebno = 10. ** (ebno_db_target / 10.)
    no = eb / ebno
    return np.sqrt(no / 2.)

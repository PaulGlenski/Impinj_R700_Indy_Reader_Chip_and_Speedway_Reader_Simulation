##
#  This is the top-level test script for the system model
##

import matplotlib.pyplot as plt
import argparse

from systems_r700.model.src.reader.reader_impairments import ReaderImpairments
from systems_r700.model.src.reader.reader_attributes import ReaderAttributes
from systems_r700.model.src.simulation_engine.simulation_engine import SimulationEngine

def run_tag():
    plt.close('all')

    attrib = ReaderAttributes()
    # make any changes to reader attributes here
    # attrib.tx_.tx_pwr_dbm_ = 30.0
    # attrib.rx_.return_loss_db_ = [-np.inf]
    # attrib.rx_.reflection_phase_ = [np.pi / 100]

    impairments = ReaderImpairments()
    # make any changes to reader impairments here

    sim_eng = SimulationEngine(link=args.link,
                               max_bits=args.max_bits,
                               packet_size=args.packet_size,
                               min_error_bits=args.min_error_bits,
                               max_error_bits=args.max_error_bits,
                               plot=args.plot,
                               save_data=args.save_ber,
                               attrib=attrib,
                               impairments=impairments)
    sim_eng.rx_ber()
    plt.show()

if __name__ == "__main__":

    _DEFAULT_LINK = 'fm0_640kHz_8LF'
    _DEFAULT_MAX_BITS = 100
    _DEFAULT_PACKET_SIZE = 10  #Packet_Size must be a multiple of 3 for 3b4b
    _DEFAULT_MIN_ERROR_BITS = 0
    _DEFAULT_MAX_ERROR_BITS = _DEFAULT_MAX_BITS
    _DEFAULT_PLOT = True
    _DEFAULT_SAVE_DATA = False

    parser = argparse.ArgumentParser(description='End-to-End r_700 BER Simulation')
    parser.add_argument('--max_bits', type=int, default=_DEFAULT_MAX_BITS, help='Total number of bits passed into the Sim, Defaults to 30')
    parser.add_argument('--packet_size', type=int, default=_DEFAULT_PACKET_SIZE, help='Size of each packet of bits to be used in the Sim, Deafults to 9')
    parser.add_argument('--min_error_bits', type=int, default=_DEFAULT_MIN_ERROR_BITS, help="Minimum number of bits that can become errors, Defaults to 0")
    parser.add_argument('--max_error_bits', type=int, default=_DEFAULT_MAX_ERROR_BITS, help="Maximum number of bits that can become errors, Defaults to 0")
    parser.add_argument('--plot', type=bool, default=_DEFAULT_PLOT, help="Boolean value indicating if the resulting BER Curve will be plotted, Defaults to True")
    parser.add_argument('--save_ber', type=bool, default=_DEFAULT_SAVE_DATA, help="Boolean value indicating if the BER of each EbNo will be saved to the BER Values Pickle, Defaults to True")
    parser.add_argument('--link', type=str, default=_DEFAULT_LINK, help='Reverse link configuration, options are:'
                                                                        'fm0_640kHz_8LF'
                                                                        'miller_4_320kHz_8LF'  
                                                                        '3b4b_baseband'
                                                                        '3b4b_subcarrier_m2'  
                                                                        'bpsk_subcarrier_m2'
                                                                        'Defaults to fm0_640kHz_8LF')

    args = parser.parse_args()
    print(args)

    run_tag()

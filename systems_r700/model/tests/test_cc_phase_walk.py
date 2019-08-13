##
#  This is the top-level test script for running the phase walk test on the carrier cancellation algorithm
##

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from systems_r700.model.src.reader.reader_impairments import ReaderImpairments
from systems_r700.model.src.reader.reader_attributes import ReaderAttributes
from systems_r700.model.src.simulation_engine.simulation_engine import SimulationEngine
import systems_r700.model.src.common.utils as ut

plt.close('all')

titlesize = 6
labelsize = 6
ticklabelsize = 6

attrib = ReaderAttributes()
impairments = ReaderImpairments()
attrib.tx_.tx_power_dbm_ = 30
step = 200
#phase_walk_range = np.linspace((np.pi/2.0), np.pi, num=step, endpoint=True)  # one quadrant
phase_walk_range = np.linspace(0, 2*np.pi, num=step, endpoint=True)  # all quadrants
#phase_walk_range = [phase_walk_range[20]]  # single phase
quantized = True  # quantized lms steps

data_dict = {}
lms_cc = []  # lms last step's residual error across phase walk
ideal_cc = []  # ideal's residual error across phase walk
rx_last = []  # last rx sample across phase walk
fine_min = []  # fine search min error across phase walk
final_min = []  # final search min error across phase walk
#fig, ax = plt.subplots(3, sharex=True)
fig, ax = ut.create_figure_doc(w=6, h=3, nrows=3, ncols=1, sharex=True)
plt.interactive(False)
for index, phase in enumerate(phase_walk_range):
    attrib.rx_.reflection_phase_ = [phase]
    sim_eng = SimulationEngine(attrib, impairments)
    cc_coarse, cc_fine, cc_lms, cc_final = sim_eng.process_cc_lms_search(quantized)

    cc_coarse_min = min(cc_coarse.keys(), key=(lambda k: cc_coarse[k]))
    print(cc_coarse_min)
    #
    # self._cc_i_sign = error_dict_final_min[0]
    # self._cc_q_sign = error_dict_final_min[1]
    # cci_float = error_dict_final_min[2]
    # self._cc_i = int(cci_float)
    # self._cc_q = error_dict_final_min[3]
    #
    fine_min.append(min(cc_fine.items(), key=lambda x: x[1])[1][0])
    final = min(cc_final.items(), key=lambda x: x[1])
    final_sign_i = final[0][0]
    final_sign_q = final[0][2]
    final_i = final_sign_i * (31.9375 - final[0][1]*0.0625)
    final_q = final_sign_q * (31.9375 - final[0][3]*0.0625)
    final_min_err = final[1]
    final_min.append(final_min_err)

    i_gains = final_sign_i * abs((sim_eng.reader_._lms_cci - 511) * 0.0625)
    q_gains = final_sign_q * abs((sim_eng.reader_._lms_ccq - 511) * 0.0625)
    lms_steps = i_gains + 1j * q_gains
    residual_steps = sim_eng.reader_._lms_residual_dc
    ideal_i, ideal_q, ideal_residual = sim_eng.calculate_cc_ideal()
    data_dict[phase] = (residual_steps[-1], lms_steps[-1].real, lms_steps[-1].imag, ideal_i, ideal_q, final_i, final_q)

    lms_cc.append(residual_steps[-1])
    rx_last.append(abs(sim_eng.reader_.rfa_.rx_[-1]))
    ideal_cc.append(abs(ideal_residual))

    ax[0].plot(residual_steps)
    solution_delta_i = np.abs(np.array(lms_steps).real) - np.abs(ideal_i)
    solution_delta_q = np.abs(np.array(lms_steps).imag) - np.abs(ideal_q)
    ax[1].plot(solution_delta_i)
    ax[2].plot(solution_delta_q)
    if abs(solution_delta_i[-1]) > 0.25 or abs(solution_delta_i[-2]) > 0.25:
        print("Phase Index " + str(index) + " did not settle within 1 step of ideal's I: " + str(
            solution_delta_i[-1]) + " and " + str(solution_delta_i[-2]))
    if abs(solution_delta_q[-1]) > 0.25 or abs(solution_delta_q[-2]) > 0.25:
        print("Phase Index " + str(index) + " did not settle within 1 step of ideal's Q: " + str(
            solution_delta_q[-1]) + " and " + str(solution_delta_q[-2]))
ax[0].set_ylabel('Residual Magnitude', fontsize=labelsize)
ax[1].set_ylabel('Relative I Position', fontsize=labelsize)
ax[2].set_ylabel('Relative Q Position', fontsize=labelsize)
ax[2].set_xlabel('LMS Iteration', fontsize=labelsize)
ax[0].tick_params(axis='both', which='major', labelsize=ticklabelsize)
ax[1].tick_params(axis='both', which='major', labelsize=ticklabelsize)
ax[2].tick_params(axis='both', which='major', labelsize=ticklabelsize)
plt.show(block=False)

lms_canc = []
ideal_canc = []
fine_canc = []
final_canc = []
hw_expectation = -40
#fig, ax = plt.subplots(1)
fig, ax = ut.create_figure_doc(w=6, h=3)
plt.interactive(False)
for sample_index in range(len(lms_cc)):
    fine_cancellation = ut.lin2db(fine_min[sample_index] / rx_last[sample_index])
    fine_canc.append(fine_cancellation)
    lms_cancellation = ut.lin2db(lms_cc[sample_index] / rx_last[sample_index])
    lms_canc.append(lms_cancellation)
    ideal_cancellation = ut.lin2db(ideal_cc[sample_index] / rx_last[sample_index])
    ideal_canc.append(ideal_cancellation)
    final_cancellation = ut.lin2db(final_min[sample_index] / rx_last[sample_index])
    final_canc.append(final_cancellation)
    # if lms_cancellation > hw_expectation:
    #     print "Phase Index has cancellation worse than -40dB: " + str(sample_index) + ', cancellation: ' + str(lms_cancellation)
ax.plot(np.arange(0, len(phase_walk_range), 1), ideal_canc, '-ro',
        np.arange(0, len(phase_walk_range), 1), fine_canc, '-mo',
        np.arange(0, len(phase_walk_range), 1), lms_canc, '-bo',
        np.arange(0, len(phase_walk_range), 1), final_canc, '-ko', markersize=3, markerfacecolor='None')
ax.plot(np.arange(0, len(phase_walk_range), 1), hw_expectation * np.ones(len(lms_canc)), 'g')
plt.legend(['Ideal Quantized', 'After Fine Search', 'After LMS', 'After Final Search'], loc='lower right')
plt.xlabel('Phase Index')
plt.ylabel('Cancellation (dB)')
plt.title('Cancellation before/after Canc Combiner')
ut.annotate_figure(title='Cancellation before/after Canc Combiner', xlabel='Phase Index', ylabel='Cancellation (dB)', ylim=[-100, 0])
plt.show(block=False)

#fig, ax = plt.subplots(1)
fig, ax = ut.create_figure_doc(w=6, h=3)
no_diff_counter = 0
plt.interactive(False)
for index, phase in enumerate(phase_walk_range):
    delta_i = data_dict[phase][1] - data_dict[phase][3]
    delta_q = data_dict[phase][2] - data_dict[phase][4]
    # if abs(delta_i) > 0.125:
    #     print "LMS Phase Index has I abs dB delta > 0.125: " + str(index) + ', delta: ' + str(delta_i) + ', cancellation: ' + str(lms_canc[index])
    # if abs(delta_q) > 0.125:
    #     print "LMS Phase Index has Q abs dB delta > 0.125: " + str(index) + ', delta: ' + str(delta_q) + ', cancellation: ' + str(lms_canc[index])
    # if delta_i == 0.0 and delta_q == 0.0:
    #     no_diff_counter += 1
    #     print "LMS Phase Index has same as Ideal: " + str(index) + ', cancellation: ' + str(lms_canc[index])
    delta_final_i = data_dict[phase][3] - data_dict[phase][5]
    delta_final_q = data_dict[phase][4] - data_dict[phase][6]
    if delta_final_i != 0.0 or delta_final_q != 0.0:
        print("Final Phase Index NOT same as Ideal: " + str(index) + ', cancellation: ' + str(lms_canc[index]))
    ax.scatter(index, delta_i, marker='o', s=10, facecolors='none', edgecolors='b', alpha=0.8, label="I" if index == 0 else "")
    ax.scatter(index, delta_q, marker='o', s=10, facecolors='none', edgecolors='r', alpha=0.8, label="Q" if index == 0 else "")
# print "Number of solutions matching Ideal: " + str(no_diff_counter)
ax.plot(np.arange(0, len(phase_walk_range), 1), 0.25*np.ones(len(phase_walk_range)), 'g')
ax.plot(np.arange(0, len(phase_walk_range), 1), -0.25*np.ones(len(phase_walk_range)), 'g')
ax.yaxis.set_minor_locator(tck.MultipleLocator(0.25))
plt.legend(loc='lower right')
plt.title('LMS last step wrt Ideal Solution')
plt.xlabel('Phase Index')
plt.ylabel('LMS vs Ideal Solution Delta')
ut.annotate_figure(title='LMS last step wrt Ideal Solution', xlabel='Phase Index', ylabel='LMS vs Ideal Solution Delta', ylim=[-1, 1])
plt.grid(True, which='major', axis='y')

plt.show()

print('done')
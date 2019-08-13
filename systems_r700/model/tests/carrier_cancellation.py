##
#  This is the top-level test script for carrier cancellation routine
#  This includes running the all search, coarse search, fine search, lms descent, and final search.
##

import numpy as np
import matplotlib.pyplot as plt
import systems_r700.model.src.common.utils as ut
from systems_r700.model.src.reader.reader_impairments import ReaderImpairments
from systems_r700.model.src.reader.reader_attributes import ReaderAttributes
from systems_r700.model.src.simulation_engine.simulation_engine import SimulationEngine
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
from matplotlib import cm

attrib = ReaderAttributes()
# make any changes to reader attributes here
attrib.tx_.tx_power_dbm_ = 30
step = 200
phase_walk_range = np.linspace(0, 2*np.pi, num=step, endpoint=True)
attrib.rx_.reflection_phase_ = [phase_walk_range[30]]

impairments = ReaderImpairments()
# make any changes to reader impairments here
# impairments.rx.temperature = 30

sim_eng = SimulationEngine(attrib, impairments)
cc_all = sim_eng.process_cc_all_search()
quantized = True
cc_coarse, cc_fine, cc_lms, cc_final = sim_eng.process_cc_lms_search(quantized)
ideal_i, ideal_q, ideal_residual = sim_eng.calculate_cc_ideal()

cc_lsb_db = 0.125
ideal_states = np.arange(31.875, 0, -cc_lsb_db)
i_word = [cc_all_sign_i * cc_all_i for cc_all_sign_i, cc_all_i, cc_all_sign_q, cc_all_q in cc_all.keys()]
q_word = [cc_all_sign_q * cc_all_q for cc_all_sign_i, cc_all_i, cc_all_sign_q, cc_all_q in cc_all.keys()]
z = [error for error in cc_all.values()]
resX = 100
resY = 100
xi = np.linspace(min(i_word), max(i_word), resX)
yi = np.linspace(min(q_word), max(q_word), resY)
X, Y = np.meshgrid(xi, yi)
Z = griddata(i_word, q_word, z, xi, yi, interp='linear')

fig = plt.figure()
fig.set_size_inches(17, 13)
plt.interactive(False)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, rstride=1, cstride=1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel('I Word (dB)')
plt.ylabel('Q Word (dB)')
plt.show(block=False)

fig = plt.figure()
plt.interactive(False)
im = plt.imshow(Z, origin='lower', extent=(-255, 255, -255, 255))
coarse_levels = np.arange(0, max(cc_coarse.values()), max(cc_coarse.values()) / 20)
CS = plt.contour(X, Y, Z, coarse_levels, linewidths=0.5)
cc_coarse_plt = {(keys[0] * keys[1], keys[2] * keys[3]): values for keys, values in cc_coarse.items()}
plt.scatter(*zip(*cc_coarse_plt.keys()))
for key, value in cc_coarse_plt.iteritems():
    plt.annotate('%.3f'%(value), xy=key, size=8)
plt.xlabel('I')
plt.ylabel('Q')
plt.axis([-255, 255, -255, 255])
plt.show(block=False)

fig = plt.figure()
plt.interactive(False)
cc_fine_plt = {(keys[0] * keys[1], keys[2] * keys[3]): values[0] for keys, values in cc_fine.items()}
fine_max_i = max(cc_fine_plt.keys(), key=lambda x: x[0])[0] + 10
fine_max_q = max(cc_fine_plt.keys(), key=lambda x: x[1])[1] + 10
fine_min_i = min(cc_fine_plt.keys(), key=lambda x: x[0])[0] - 10
fine_min_q = min(cc_fine_plt.keys(), key=lambda x: x[1])[1] - 10
im = plt.imshow(Z, origin='lower', extent=(-255, 255, -255, 255))
fine_levels = np.arange(0, 0.5, 0.5 / 20)
CS = plt.contour(Z, fine_levels, origin='lower', extent=(-255, 255, -255, 255), linewidths=0.5)
plt.scatter(*zip(*cc_fine_plt.keys()))
for key, value in cc_fine_plt.iteritems():
    plt.annotate('%.5f'%(value), xy=key, size=8, color='m')
cc_lms_plt = {(keys[0] * keys[1], keys[2] * keys[3]): values[0] for keys, values in cc_lms.items()}
plt.scatter(*zip(*cc_lms_plt.keys()))
for key, value in cc_lms_plt.iteritems():
    plt.annotate('%i'%(value), xy=key, size=8, color='g')
cc_final_plt = {(keys[0] * keys[1], keys[2] * keys[3]): values for keys, values in cc_final.items()}
plt.scatter(*zip(*cc_final_plt.keys()))
for key, value in cc_final_plt.iteritems():
    plt.annotate('%.5f'%(value), xy=key, size=8, color='y')
ideal_canc = (np.sign(ideal_i) * np.where(ideal_states == abs(ideal_i))[0][0], np.sign(ideal_q) * np.where(ideal_states == abs(ideal_q))[0][0])
plt.scatter(ideal_canc[0], ideal_canc[1])
plt.annotate('Ideal', xy=ideal_canc, size=8, color='r', xytext=(ideal_canc[0], ideal_canc[1]-0.1))
plt.xlabel('I')
plt.ylabel('Q')
plt.axis([fine_min_i, fine_max_i, fine_min_q, fine_max_q])
plt.show(block=False)

fig, ax = plt.subplots(4, sharex=True)
plt.interactive(False)
ax[0].plot(sim_eng.reader_._dci_updatei)
ax[1].plot(sim_eng.reader_._dcq_updatei)
ax[2].plot(sim_eng.reader_._dci_updateq)
ax[3].plot(sim_eng.reader_._dcq_updateq)
ax[0].set_ylabel('DCII')
ax[1].set_ylabel('DCQI')
ax[2].set_ylabel('DCIQ')
ax[3].set_ylabel('DCQQ')
plt.show(block=False)

fig, ax = plt.subplots(5, sharex=True)
plt.interactive(False)
fine_min = min(cc_fine.items(), key=lambda x: x[1][0])
lms_lin_i = np.array(sim_eng.reader_._lms_i_list)
lms_lin_q = np.array(sim_eng.reader_._lms_q_list)
lms_steps = np.array(sim_eng.reader_._lms_steps)
residual_steps = np.array(sim_eng.reader_._residual_steps)
residual_steps = np.insert(residual_steps, 0, fine_min[1])
upd1 = np.array(sim_eng.reader_._upd1)
upd2 = np.array(sim_eng.reader_._upd2)
cancellation = np.array(ut.lin2db(sim_eng.reader_._residual_steps / abs(sim_eng.reader_.rfa_.rx_[-1])))
cancellation = np.insert(cancellation, 0, ut.lin2db(fine_min[1] / abs(sim_eng.reader_.rfa_.rx_[-1])))
ax[0].plot(sim_eng.reader_._weighted_mu, '-bo')
ax[1].plot(lms_steps.real, '-bo')
ax[2].plot(lms_steps.imag, '-bo')
ax[3].plot(residual_steps, '-bo')
ax[4].plot(cancellation, '-bo')
ax[0].set_ylabel('Weighted Mu')
ax[0].grid(which='major', linestyle='--', linewidth=0.5)
ax[1].set_ylabel('I After Update')
ax[1].grid(which='major', linestyle='--', linewidth=0.5)
ax[2].set_ylabel('Q After Update')
ax[2].grid(which='major', linestyle='--', linewidth=0.5)
ax[3].set_ylabel('Residual Error Mag')
ax[3].grid(which='major', linestyle='--', linewidth=0.5)
ax[4].set_ylabel('Cancellation (dB)')
ax[4].grid(which='major', linestyle='--', linewidth=0.5)
plt.show(block=False)

print('done')
from kymatio.scattering1d.filter_bank import scattering_filter_factory
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.rcParams.update(mpl.rcParamsDefault)
N = 2**13
J = 5
Q = (8, 1)
phi_f, psi1_f, psi2_f = scattering_filter_factory(N, J, Q, 2**5)
print(1, len(psi1_f), len(psi2_f))

plt.figure()

plt.subplot(211)
plt.plot(np.arange(T)/T,phi_f['levels'][0], 'r')
for psi_f in psi1_f:
    plt.plot(np.arange(T)/T, psi_f['levels'][0], 'b')
    
plt.xlim(0, 0.7)

#plt.xlabel(r'$\omega$', fontsize=18)
plt.ylabel(r'$\hat\phi(\omega), \, \hat\psi_{\lambda_1}(\omega)$', fontsize=18)



plt.subplot(212)
for psi_f in psi2_f:
    plt.plot(np.arange(T)/T, psi_f['levels'][0], 'b')

plt.xlim(0, 0.7)

plt.xlabel(r'$\omega$', fontsize=18)
plt.ylabel(r'$\hat\psi_{\lambda_2}(\omega)$', fontsize=18)
#plt.title('Frequency response of second-order filters (Q = {})'.format(Q[1]),
 #         fontsize=12)
plt.show()


psi_time = np.fft.ifft(psi1_f[-1]['levels'][0])
psi_real = np.real(psi_time)
psi_imag = np.imag(psi_time)
plt.plot(np.concatenate((psi_real[-2**8:],psi_real[:2**8])),'b')
plt.plot(np.concatenate((psi_imag[-2**8:],psi_imag[:2**8])),'r')
#plt.plot(psi_real)
#plt.plot(psi_imag)

plt.xlabel(r'$t$', fontsize=18)
plt.ylabel(r'$\psi(t)$', fontsize=18)
plt.title('First-order filter - Time domain (Q = {})'.format(Q), fontsize=12)
plt.legend(["$\psi$_real","$\psi$_imag"])
plt.show()

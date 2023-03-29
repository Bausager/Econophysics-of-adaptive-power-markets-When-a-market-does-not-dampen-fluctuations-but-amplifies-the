import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import os
from time import sleep
import time # For timing and waiting.
import pandas as pd


from multiprocessing import Process
from multiprocessing import cpu_count


# https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
def acf(x, length=20):
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  \
        for i in tqdm(range(1, length))])


def priceSeries(T, T_init, v_0, sigma_0, mu=1):
    

    P = np.arange(T, dtype='float')
    P_temp = 0.0
    
    for t in range(0, int(T_init)):
        if(t == 0):
            P_temp = P_temp + (-v_0*(np.random.normal(loc=mu) - mu) + (sigma_0*np.random.normal(loc=mu)))
        else:
            P_temp = P_temp + (-v_0*(P_temp - mu) + (sigma_0*np.random.normal(loc=mu)))
        
        
    for i in tqdm(range(0, int(T))):
        
        if(i == 0):
            P[i] = P_temp + (-v_0*(P_temp - mu) + (sigma_0*np.random.normal(loc=mu)))
        else:
            P[i] = P[i-1] + (-v_0*(P[i-1] - mu) + (sigma_0*np.random.normal(loc=mu)))
            
    return P


def generate(P,
            T, 
            N,
            f,
            bins,
            core,
            N_cores):

    
    # For timing
    start = 0
    ETA = 0

    # Bincounter for acceptable price density
    acceptable_price_density = np.zeros(len(bins)-1)
    # Bincounter for load price density
    load_price_density = np.zeros(len(bins)-1)
    
    p = np.zeros(shape=(int(N)), dtype=float)
    p = np.random.uniform(low=0.0, high=1.0, size=int(N))
    d = np.zeros(shape=(int(N)), dtype=float)

    D = np.zeros(shape=int(T), dtype=float)

    if (core == 0):
        pi = np.zeros(shape=(int(T)), dtype=float)
        di = np.zeros(shape=(int(T)), dtype=float)

    for i in (range(1, int(T))):

        if (i%10e3==0):
            
            # Measure end time.
            end = time.perf_counter()

            ETA = ((end-start)/10E3)*(T-i)
            H = int(np.floor(ETA/60/60))
            M = int(round(ETA/60)%60)
            S = round(ETA%60)
            print(f'Core[{core}]: [{i}/{T}], ETA: {H}:{M}:{S}')

            # Measure start time.
            start = time.perf_counter()

        d = np.where(P[i] <= p, 1, 0)
        d_inv = np.where(d == 0, 1, 0)
        d2 = d_inv*np.random.binomial(n=1, p=f, size=int(N))
        d2_inv = d_inv*np.where(d2 == 0, 1, 0) 
        
        if(core == 0):
            pi[i] = p[0]
            di[i] = d[0]
        
        p1 = d*(np.random.uniform(low=0.0, high=p))
        p2 = d2*(np.random.uniform(low=p, high=1.0))
        p3 = d2_inv*p
        
        p = (p1 + p2 + p3)

        D[i] = np.sum(d)

        acceptable_price_density += (np.histogram(p, bins)[0])
        load_price_density += (np.histogram(P[i], bins)[0])*D[i]

    if (core == 0):
        np.save('./Storage/pi', pi)
        np.save('./Storage/di', di)

    np.save(f'./Storage/D{core}', D)
    np.save(f'./Storage/acceptable_price_density{core}', acceptable_price_density)
    np.save(f'./Storage/load_price_density{core}', load_price_density)



if __name__ == '__main__':

    # Cores used in the script
    #N_cores = 1
    N_cores = cpu_count()-1
    
    # Number of Agents per core (ish)
    N = 1e6

    M = int(np.floor(N/N_cores))
    N = M*N_cores

    # Time series lenght
    T = 1E7

    # probability for agent to change price per step
    f = 1E-3
    
    
    # Price series (Time series)
    P = np.zeros(shape=int(T), dtype=float)
    # Demand series
    D = np.zeros(shape=int(T), dtype=float)

    pi = np.zeros(shape=(int(T)), dtype=float)
    di = np.zeros(shape=(int(T)), dtype=float)
    

    bin_size = 100
    # Bins for histogram
    bins = np.linspace(start=0, stop=2, num=bin_size)
    # Bincounter for acceptable price density
    acceptable_price_density = np.zeros(bin_size-1)
    # Bincounter for load price density
    load_price_density = np.zeros(bin_size-1)
    # Bincounter for price series
    price_density = np.zeros(bin_size-1)

    # Create time series if not found.
    if not (os.path.exists(f'./Storage/P{T}.npy') and os.path.exists(f'./Storage/N{N}.npy') and os.path.exists(f'./Storage/f{f}.npy')):
        P = priceSeries(T=T, T_init=int(1E3), v_0=0.2, sigma_0=0.1)

        processes = [Process(target=generate, args=(P, T, M, f, bins, i, N_cores)) for i in range(0,N_cores)]
        # Start all processes.
        for process in processes:
                process.start()
        # Wait for all processes to complete.
        for process in processes:
            process.join()

        for core in range(0, N_cores):
            D += np.load(f'./Storage/D{core}.npy')
            acceptable_price_density += np.load(f'./Storage/acceptable_price_density{core}.npy')
            load_price_density += np.load(f'./Storage/load_price_density{core}.npy')

            os.remove(f'./Storage/D{core}.npy')
            os.remove(f'./Storage/acceptable_price_density{core}.npy')
            os.remove(f'./Storage/load_price_density{core}.npy')


        np.save(f'./Storage/P{T}', P)
        np.save(f'./Storage/N{N}', N)
        np.save('./Storage/T', T)
        np.save(f'./Storage/f{f}', f)
        np.save('./Storage/P', P) 
        np.save('./Storage/D', D)
        np.save('./Storage/bins', bins)
        np.save('./Storage/acceptable_price_density', acceptable_price_density)
        np.save('./Storage/load_price_density', load_price_density)


    P = np.load(f'./Storage/P{T}.npy')
    N = np.load(f'./Storage/N{N}.npy')
    T = np.load('./Storage/T.npy')
    f = np.load(f'./Storage/f{f}.npy')
    pi = np.load('./Storage/pi.npy')
    di = np.load('./Storage/di.npy')
    D = np.load('./Storage/D.npy')

    D_bar = np.average(D)
    d_bar = D_bar / N

    bins = np.load('./Storage/bins.npy')
    delta_bins = np.zeros(len(bins)-1)
    delta_bins = bins[1:] - bins[:-1]
    bins_x = (bins[1:] + bins[:-1])/2

    acceptable_price_density = np.load('./Storage/acceptable_price_density.npy')
    acceptable_price_density = acceptable_price_density/(np.sum((acceptable_price_density*delta_bins)))
    load_price_density = np.load('./Storage/load_price_density.npy')
    load_price_density = load_price_density/(np.sum((load_price_density*delta_bins)))
    price_density = np.histogram(P, bins,  density=True)[0]



###
### Figures
###

    plt.figure(1, figsize=(16,9))
    plt.title('FIG. 1')
    plt.plot(P[:5000], color='b', label="Price")
    plt.plot(pi[:5000], color='k', label="Price acceptance")
    plt.plot(di[:5000], color='r', label="Demand")
    plt.legend(loc='upper right')
    plt.xlabel("Time")
    plt.grid()
    plt.tight_layout()
    plt.savefig('FIG. 1.png')
#
    fig, axs = plt.subplots(2, figsize=(16,9))
    fig.suptitle('FIG. 2')
    axs[0].plot(P[:6500])
    axs[0].set_ylabel("P")
    axs[1].plot(D[:6500]/D_bar, color='r')
    axs[1].set_ylabel(r'$D/\bar{D}$')
    axs[1].set_xlabel(r'$t$')
    axs[0].grid()
    axs[1].grid()
    fig.tight_layout()
    plt.savefig('FIG. 2.png')
#
    plt.figure(3, figsize=(16,9))
    plt.title("FIG. 3")
    plt.plot(bins_x, acceptable_price_density, "g-", label="Price acceptance density")
    plt.plot(bins_x, load_price_density, "b-", label="loads consumed density")
    plt.plot(bins_x, price_density, "r-", label="Price density")
    plt.xlim(0, 1.3)
    plt.ylim(0, np.max(price_density)*1.1)
    plt.xlabel(r'$P$')
    plt.ylabel(r'$\rho$')
    plt.axvline(x = np.average(P), color = 'k')
    plt.axvline(x = 1+(1*np.std(P)), color = 'k', linestyle='dashed', alpha=0.35)
    plt.axvline(x = 1-(1*np.std(P)), color = 'k', linestyle='dashed', alpha=0.35)
    plt.axvline(x = 1-(2*np.std(P)), color = 'k', linestyle='dashed', alpha=0.35)
    plt.axvline(x = 1-(3*np.std(P)), color = 'k', linestyle='dashed', alpha=0.35)
    plt.axvline(x = 1-(4*np.std(P)), color = 'k', linestyle='dashed', alpha=0.35)
    plt.axvline(x = 1-(5*np.std(P)), color = 'k', linestyle='dashed', alpha=0.35)
    plt.legend(loc='upper right')
    plt.savefig('FIG. 3.png')
#
    plt.figure(4, figsize=(16,9))
    plt.title('FIG. 4')
    # Bins for histogram
    bins = np.linspace(start=0.01, stop=3000, num=250)
    # Use non-equal bin sizes, such that they look equal on log scale.
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    # Bincounter for acceptable price density
    Demand_density = (np.histogram(D/D_bar, logbins, density=True)[0])

    new_x = (logbins[1:] + logbins[:-1])/2

    plt.plot(new_x, Demand_density, label=f'N={int(N)}, T={int(T)}, f={f}')
    plt.axvline(x = 1, color = 'k')
    plt.ylim(1E-8, 1E0)
    plt.yticks([1E-8, 1E-7, 1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1E0, 1E1])
    plt.xlim(1E-1, 1E3)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$\rho$')
    plt.xlabel(r'$D/\bar{D}$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('FIG. 4.png')
#
    plt.figure(5, figsize=(16,9))
    plt.title('FIG. 5')
    bins = np.linspace(start=0.01, stop=3000, num=100)
    # Use non-equal bin sizes, such that they look equal on log scale.
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist2d(P, D/D_bar, bins=(100, logbins), norm=mpl.colors.LogNorm(), cmap='YlOrRd')
    plt.colorbar()
    plt.axis([0.35, 1.01, 1E-2, 1E3])
    plt.ylabel(r'$D/\bar{D}$')
    plt.xlabel(r'$P$')
    plt.yscale("log")
    plt.grid()
    plt.tight_layout()
    plt.savefig('FIG. 5.png')



###
### Import of file
###
    all_real = pd.read_csv('EnergyReport.csv', delimiter=';', decimal=',')
    #all_real = all_real[all_real['DK1_DKK/MWh'] > 0.0]
    N = len(all_real)
###
### Declarations
###
    P_real = np.zeros(N, dtype=float)
    P_wo_mean = np.zeros(N, dtype=float)
    deltaP = np.zeros(N-1, dtype=float)
    matr =  np.zeros([N-1, 2], dtype=float)
###
### Fit data to Langevin-equation
###
    P_real = all_real['DK1_DKK/MWh'].to_numpy()
    deltaP = P_real[1:] - P_real[:-1]
    P_mean = np.mean(P_real)
    P_wo_mean = P_real - P_mean
    Normal_Distribution = np.random.normal(loc=P_mean, size=N-1)

    matr[:,0] = P_wo_mean[:-1]
    matr[:,1] = Normal_Distribution

    [v_0, sigma_0] = np.dot(np.linalg.pinv(matr), deltaP)
    print(f'v_0: {v_0}, sigma_0: {sigma_0}, P_mean: {P_mean}')
###
### Generate price series based on real data
###
    P_new = priceSeries(T=T, T_init=int(1E3), v_0=-v_0, sigma_0=sigma_0, mu=P_mean)

    P_acorr = acf(P, length=int(250))
    P_new_acorr = acf(P_new, length=int(250))
    

    # Bins for histogram
    bins = np.linspace(start=0, stop=750, num=250)
    price_density = np.histogram(all_real['DK1_DKK/MWh'], bins,  density=True)[0]
    bins_x = np.zeros(len(bins)-1)
    bins_x = (bins[1:] + bins[:-1])/2

    plt.figure(6, figsize=(16,9))
    plt.title("Real Price PDF")
    plt.plot(bins_x, price_density)
    plt.grid()
    plt.savefig('FIG. 6 - PDF of real price series.png')
    






    fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(16,9))
    ax1.title.set_text('Autocorrelation of paper price series')
    ax1.plot(P_acorr)
    ax1.grid(True)

    ax2.title.set_text('Autocorrelation of real-based price series')
    ax2.plot(P_new_acorr)
    ax2.grid(True)
    plt.savefig('FIG. 7 - Autocorrelations of price series.png')




    plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import os
from time import sleep
import time # For timing and waiting.

from multiprocessing import Process
from multiprocessing import cpu_count


def priceSeries(T, T_init, v_0, sigma_0, mu=1):
    

    P = np.arange(T, dtype='float')
    P_temp = 0.0
    
    for t in range(0, int(T_init)):
        if(t == 0):
            P_temp = P_temp + (-v_0*(np.random.normal(loc=1) - mu) + (sigma_0*np.random.normal()))
        else:
            P_temp = P_temp + (-v_0*(P_temp - mu) + (sigma_0*np.random.normal()))
        
        
    for i in tqdm(range(0, int(T))):
        
        if(i == 0):
            P[i] = P_temp + (-v_0*(P_temp - mu) + (sigma_0*np.random.normal()))
        else:
            P[i] = P[i-1] + (-v_0*(P[i-1] - mu) + (sigma_0*np.random.normal()))
            
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
    
    # Bins for histogram
    bins = np.linspace(start=0, stop=2, num=100)
    # Bincounter for acceptable price density
    acceptable_price_density = np.zeros(len(bins)-1)
    # Bincounter for load price density
    load_price_density = np.zeros(len(bins)-1)
    # Bincounter for price series
    price_density = np.zeros(len(bins)-1)

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
    # density of highest acceptable prices scale
    scale = 1.0 / (N*(T+1))
    D = np.load('./Storage/D.npy')
    D_bar = np.average(D)
    d_bar = D_bar / N
    bins = np.load('./Storage/bins.npy')
    acceptable_price_density = np.load('./Storage/acceptable_price_density.npy')
    load_price_density = np.load('./Storage/load_price_density.npy')
    pi = np.load('./Storage/pi.npy')
    di = np.load('./Storage/di.npy')

    price_density = (np.histogram(P, bins,  density=True)[0])



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
    fig, ax = plt.subplots(figsize=(20,9))
    fig.suptitle('FIG. 3')
    fig.subplots_adjust(right=0.75)
    twin1 = ax.twinx()
    twin2 = ax.twinx()
    twin2.spines.right.set_position(("axes", 1.08))

    new_x = (bins[1:] + bins[:-1])/2

    p1, = ax.plot(new_x, acceptable_price_density*(scale), "g-", label="Price acceptance density")
    p2, = twin1.plot(new_x, load_price_density/D_bar, "b-", label="loads consumed density")
    p3, = twin2.plot(new_x, price_density, "r-", label="Price density")
    ax.set_xlim(0, 1.3)
    ax.set_ylim(0, np.max(acceptable_price_density*(scale))*1.1)
    twin1.set_ylim(0, np.max(load_price_density/D_bar)*1.1)
    twin2.set_ylim(0, np.max(price_density)*1.1)
    ax.set_xlabel(r'$P$')
    ax.set_ylabel(r'$\rho$')
    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.axvline(x = np.average(P), color = 'k')
    ax.axvline(x = 1+(1*np.std(P)), color = 'k', linestyle='dashed', alpha=0.35)
    ax.axvline(x = 1-(1*np.std(P)), color = 'k', linestyle='dashed', alpha=0.35)
    ax.axvline(x = 1-(2*np.std(P)), color = 'k', linestyle='dashed', alpha=0.35)
    ax.axvline(x = 1-(3*np.std(P)), color = 'k', linestyle='dashed', alpha=0.35)
    ax.axvline(x = 1-(4*np.std(P)), color = 'k', linestyle='dashed', alpha=0.35)
    ax.axvline(x = 1-(5*np.std(P)), color = 'k', linestyle='dashed', alpha=0.35)
    ax.tick_params(axis='x', **tkw)
    ax.legend(handles=[p1, p2, p3], loc='upper right')
    fig.savefig('FIG. 3.png')
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
    plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
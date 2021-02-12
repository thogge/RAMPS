import numpy as np
from scipy import interpolate
import random
import matplotlib as mpl
#mpl.use('Agg')
mpl.rc('xtick', labelsize=20) 
mpl.rc('ytick', labelsize=20) 
import matplotlib.pyplot as plt
import sys,os,getopt
import multiprocessing as mp
import pdb

def k(xi,xj,tau,l):
    return tau*np.exp(-0.5*((xi-xj)/l)**2)

def gauss_proc(ti,tt,y,E,tau,l):
    E_mu = np.zeros(len(tt))
    var_mu = np.zeros(len(tt))
    tt1,tt2 = np.meshgrid(ti,ti)
    for i,t in enumerate(tt):
        E_mu[i] = k(ti,t,tau,l).dot(np.linalg.inv(k(tt1,tt2,tau,l)+E)).dot(y)
        var_mu[i] = k(t,t,tau,l)-k(t,ti,tau,l).dot(np.linalg.inv(k(tt1,tt2,tau,l)+E)).dot(k(ti,t,tau,l))
    return E_mu,var_mu

def acf(xt,h):
    N = xt.shape[0]
    xt1 = xt[0:N-h]
    xt2 = xt[h:N]
    m = np.mean(xt)
    num = np.sum((xt1-m)*(xt2-m))
    den = np.sqrt(np.sum((xt1-m)**2)*np.sum((xt2-m)**2))
    rho = num/den
    return rho


def q(Xt,sig_Xt):
    Xtp1 = []
    for i in np.arange(len(Xt)):
        if sig_Xt[i]==0:
            Xtp1.append(random.randint(0, 1))
        else:
            Xtp1.append(np.random.normal(loc=Xt[i],scale=sig_Xt[i]))
    return Xtp1

def accept(logr):
    if logr > 0.:
        a = True
    elif logr < 0. and np.isfinite(logr):
        if np.random.uniform() <= np.exp(logr):
            a = True
        else:
            a = False
    else:
        a = False
    return a

def get_model_values(emu,vmu,t,tt):
    yg = np.zeros(len(t))
    vg = np.zeros(len(t))
    for i in range(len(t)):
        ii = np.where(abs(t[i]-tt) == np.min(abs(t[i]-tt)))[0][0]
        yg[i] = emu[ii]
        vg[i] = vmu[ii]
    return yg,vg

def get_gauss_proc(t,yi,sigs,qs,tau,l):
    tmax = np.ceil(np.max(t))
    tmin = np.floor(np.min(t))
    tt = np.linspace(tmin,tmax,(tmax-tmin))
    #pdb.set_trace()
    gp_sigs = np.copy(sigs)
    gp_sigs[np.where(qs == 0)] = sigs.max()*100.
    #pdb.set_trace()
    E = np.diag(gp_sigs)
    emu,vmu = gauss_proc(t,tt,yi,E,tau,l)
    """
    plt.fill_between(tt,emu+np.sqrt(vmu),emu-np.sqrt(vmu),zorder=0)
    plt.errorbar(t,yi,yerr=sigs,fmt='o',zorder=1)
    plt.scatter(t,yi,c=qs,zorder=2)
    plt.show()
    """
    return tt,emu,vmu

def mm_posterior(pars,yi,sigs,t):
    Pb = pars[0]
    #yg = pars[1]
    #vg = pars[2]
    yb = pars[1]
    vb = pars[2]
    qs = pars[3:]
    """
    w = np.where(yi<0.)
    print w
    #pdb.set_trace()
    if len(w[0])>0:
        qs[w[0]] = 0 
    """
    ps = np.zeros(len(sigs))
    tau = 60.
    l = 60.
    tt,emu,vmu = get_gauss_proc(t,yi,sigs,qs,tau,l)
    yg,vg = get_model_values(emu,vmu,t,tt)
    #pdb.set_trace()
    for i in np.arange(len(sigs)):
        ps[i] = np.log((1-Pb)**qs[i]*(Pb**(1-qs[i]))*((1/((2*np.pi*(sigs[i]**2+vg[i]))**0.5))*(np.exp(-0.5*(yi[i]-yg[i])**2/(sigs[i]**2+vg[i])))**qs[i])*(((1/((sigs[i]+vb)*(2*np.pi*(sigs[i]**2+vb**2))**0.5))*np.exp(-0.5*(yi[i]-yb)**2/(sigs[i]**2+vb**2)))**(1-qs[i])))
        #ps[i] = np.log((1-Pb)**qs[i]*(Pb**(1-qs[i]))*((1/((sigs[i]+vg)*(2*np.pi)**0.5))*(np.exp(-0.5*(yi[i]-yg)**2/(sigs[i]+vg)**2))**qs[i])*(((1/((sigs[i]+vb)*(2*np.pi)**0.5))*np.exp(-0.5*(yi[i]-yb)**2/(sigs[i]+vb)**2)-(1/((sigs[i]+vg)*(2*np.pi)**0.5))*(np.exp(-0.5*(yi[i]-yg)**2/(sigs[i]+vg)**2)))**(1-qs[i])))
    p = np.sum(ps)
    return p

def mcmc(N,NW,x0,step_size,yi,sigs,t,nn):
    xt_grid = np.zeros(shape=(N+1,len(x0),NW))
    lps_grid = np.zeros(shape=(N+1,NW))
    for w in range(NW):
        xt_grid[0,:,w] = x0
        lps_grid[0,w] = mm_posterior(x0,yi,sigs,t)
    for i in np.arange(N):
        if i%10 == 0:
            print float(i)/N
        #pdb.set_trace()
        out = mp.Queue()
        ps = []
        for w in range(NW):
            #pdb.set_trace()
            if np.sum(xt_grid[i,:,w])==0:
                pdb.set_trace()
            y = q(xt_grid[i,:,w],step_size)
            #print y
            #pdb.set_trace()
            ps.append(mp.Process(target=walk_once,
                                 args=(xt_grid[i,:,w],
                                       y,t,yi,sigs,out,w)))
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        results = [out.get() for p in ps]
        for n,r,l in results:
            #pdb.set_trace()
            xt_grid[i+1,:,n] = r
            lps_grid[i+1,n] = l

    w = np.where(lps_grid==np.max(lps_grid))
    #pdb.set_trace()
    test = xt_grid[w[0][0],:,w[1][0]]
    print test
    fig,ax = plt.subplots(2,1,sharex=True)
    for i in range(NW):
        ax[0].plot(xt_grid[:,0,i])
    ax[0].set_ylabel(r'$P_b$',fontsize=20)
    for i in range(NW):
        ax[1].plot(lps_grid[:,i])
    ax[1].set_ylabel('log(Posterior)',fontsize=20)
    ax[1].set_xlabel('Steps',fontsize=20)
    plt.tight_layout()
    plt.show()
    pdb.set_trace()
    """
    tau = 60.
    l = 60.
    tt,emu,vmu = get_gauss_proc(t,yi,sigs,test[3:],tau,l)
    plt.errorbar(t,yi,yerr=sigs,color='k',fmt='o',zorder=1)
    plt.fill_between(tt,emu+np.sqrt(vmu),emu-np.sqrt(vmu),zorder=0)
    plt.plot(tt,emu,color='k',zorder=2)
    plt.scatter(t,yi,c=test[3:],zorder=3)
    plt.colorbar()
    plt.xlabel('Date [MJD]',fontsize=20)
    plt.ylabel('Gain',fontsize=20)
    plt.tight_layout()
    #plt.ylim(-3,3)
    plt.savefig('mix_plot_'+str(nn)+'.png')
    plt.clf()
    #plt.show()
    #"""
    #pdb.set_trace()
    return xt_grid,lps_grid

def walk_once(xt,y,t,yi,sigs,out,pos):
    if np.logical_or(np.logical_or(y[0]<0.,y[0]>1.),y[2]<2.5):
        lr = -np.inf
    else:
        lr = mm_posterior(y,yi,sigs,t) - mm_posterior(xt,yi,sigs,t)        
    if accept(lr):
        next = y
        lps = mm_posterior(y,yi,sigs,t)
    else:
        next = xt
        lps = mm_posterior(xt,yi,sigs,t)
    out.put((pos,next,lps))


bmn = 3
ifn = 8
pln = 1
gain_file = 'gains_bm'+str(bmn)+'_if'+str(ifn)+'_pol'+str(pln)+'.txt'
err_file = 'gain_terrs_bm'+str(bmn)+'_if'+str(ifn)+'_pol'+str(pln)+'.txt'
time_file = 'gaincal_times.txt'
g = np.loadtxt(gain_file)
e = np.loadtxt(err_file)
t = np.loadtxt(time_file)
w = np.where(e>0.)
g = g[w]
e = e[w]
t = t[w]
ws = [np.where(t<57600),np.where(np.logical_and(t>57600,t<57950)),np.where(t>57950)]
#ws = [np.where(np.logical_and(t>58035,t<58090))]
N = int(10**4)
NW = 16
npars = 3
for i,s in enumerate(ws):
    print i
    #s = ws[2]
    ng=g[s]
    ne=e[s]
    nt=t[s]
    qs = np.ones(len(ng))
    #wb = np.where(np.logical_or(ng<0.5,ng>1.2))
    #qs[wb] = 0
    x0 = np.ones(len(ng)+npars)
    x0[0] = 0.1
    x0[1] = 0.
    x0[2] = 10.
    x0[3:] = qs
    step_size = np.zeros(len(x0))
    step_size[0] = 0.01
    step_size[1] = 0.1
    step_size[2] = 1
    """
    d = np.diff(g)
    plt.scatter(np.diff(t),np.diff(g),c=g[1:],cmap='Vega20c')
    #plt.plot(np.diff(t),np.diff(g),'o')
    plt.colorbar()
    plt.show()
    """
    #pdb.set_trace()
    xts,lps = mcmc(N,NW,x0,step_size,ng,ne,nt,i)
    #pdb.set_trace()
    xts_outfile = 'xts_'+str(i)+'.txt'
    lps_outfile = 'lps_'+str(i)+'.txt'
    with file(xts_outfile, 'w') as outfile:
        for w in range(NW):
            np.savetxt(xts_outfile,xts[:,:,w])
            outfile.write('\n')
    with file(lps_outfile, 'w') as outfile:
        for w in range(NW):
            np.savetxt(lps_outfile,lps[:,w])
            outfile.write('\n')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import shutil
import pickle
from datetime import datetime, date
from scipy.optimize import minimize
from scipy import interpolate

#Making sure correct directories are in place
#Run these checks outside of fxns to reduce overhead

if os.path.exists('../images') == False:
    print("...")
    print("NOTICE: ../images directory not found! Creating new ./images directory for plots")
    print("...")
    os.mkdir('../images')
    
if os.path.exists('../measurement_data') == False:
    print("...")
    print("NOTICE: ../measurement_data directory not found! Creating new ./measurement_data directory for csv files")
    print("...")
    os.mkdir('../measurement_data')

    if os.path.exists("../measurement_data/backups")!=True:
        print("No folder for parameter backups found! Creating one now...")
        os.mkdir("../measurement_data/backups")

now = datetime.now()
savepath = "../measurement_data/backups/"+now.strftime("%m%d%Y")+"parameters.pickle"
if os.path.exists(savepath)!=True:
    print("No parameters backup found for today! Creating one now...")
    shutil.copyfile("../measurement_data/parameters.pickle", savepath)


#Load data for a *specific qubit* from the master parameters.pickle file
def loadfrompickle(qubit_name, path=None):
    if path == None: path = "../measurement_data/parameters.pickle"
    with open(path, "rb") as f:
        depickled = pickle.load(f)
        qubit_config = depickled[qubit_name]
        return qubit_config

#write data for a *specific qubit* to the master parmaters.pickle file
def writetopickle(qubit_name, path=None, **kwargs):
    if path == None: path = "../measurement_data/parameters.pickle"
    with open(path, "rb") as f:
        depickled = pickle.load(f)
    for key, value in kwargs.items():
            depickled[qubit_name][key] = value
    with open(path, "wb") as f:
        pickle.dump(depickled, f)

#Dump an entire dictioionary to a specified pickle file
def dumptopickle(path, paramdict):
    with open(path, "wb") as f:
        pickle.dump(paramdict, f)

#Plot and save data for 1D graph
def plotsave1d(plot, save, q_name, title, trial, xvar, xvarname, ydata, ydataname, config, data2save=None, fitfunc=None, **kwargs):

        """
        generic plotting function for 1 sweep variable + data variable
        
        :param q_name: name of the qubit in dict_cfg
        :param plot: if True, shows a plot
        :param save: if True, saves data as .csv file, and if plot is also True, saves a .pdf image as well
        :param title: string with the name of the experiment (e.g. Length Rabi)
        :param trial: additional information in the name of the experiment to be saved, if none, it's '01'
        :param xvar: the variable on the x axis
        :param xvarname: name of the x variable to be shown on the plot
        :param ydata: the actual data to be plotted
        :param ydataname: name of y data to be shown on the plot
        :param fitfunc: fit function to be plotted over the data, needs to be an array with the same length as xvar
        :param **kwargs: any additional information you want to be saved into the .csv file
            (e.g. Reps: "200", Shape: "square pulse")
            
        """

        
        if plot is not None and plot:
            plt.plot(xvar, ydata, 'o-')
            if fitfunc is not None:
                plt.plot(xvar, fitfunc, 'k--')
            plt.xlabel(xvarname)
            plt.ylabel(ydataname)
            plt.title(title)
            
        if save is not None and save:
                if data2save is None:
                    data2save=ydata
                else:
                    if data2save.any() is None:
                        data2save=ydata
                today=date.today()
                if trial is None:
                    trial="01"
                if plot is not None and plot:
                    if os.path.exists('../images/{}'.format(q_name)) == False:
                        os.mkdir('../images/{}'.format(q_name))
                    plt.savefig('../images/{}/Stark-{}-{}-{}.pdf'.format(q_name, title, today, trial))
                    
                if os.path.exists('../measurement_data/{}'.format(q_name)) == False:
                        os.mkdir('../measurement_data/{}'.format(q_name))
                
                df=pd.DataFrame(data2save, index=xvar)
                df.to_csv('../measurement_data/{}/Data-Stark-{}-{}-{}.csv'.format(q_name, title, today, trial), mode='a', header=False)
                
                datapath = "../measurement_data/{}/Settings-Stark-{}-{}-{}.pickle".format(q_name, title, today, trial)
                for key, value in kwargs.items():
                    config[key] = value
                dumptopickle(datapath, config)

#Plot and save data for 2D graph
def plotsave2d(plot, save, q_name, title, trial, xvar, xvarname, yvar, yvarname, zdata, zdataname, config, data2save=None, **kwargs):
        
        """
        generic plotting function for 2 sweep variables + data variable
        
        :param q_name: name of the qubit in dict_cfg
        :param plot: if True, shows a plot
        :param save: if True, saves data as .csv file, and it plot is also True, as a .pdf image
        :param title: string with the name of the experiment (e.g. Length Rabi)
        :param trial: additional information in the name of the experiment to be saved, if none, it's '01'
        :param xvar: the variable on the x axis
        :param xvarname: name of the x variable to be shown on the plot
        :param yvar: the variable on the y axis
        :param yvarname: name of the y variable to be shown on the plot
        :param zdata: the actual data to be plotted
        :param zdataname: name of z data to be shown on the plot
        :param **kwargs: any additional information you want to be saved into the .csv file
            (e.g. Reps: "200", Shape: "square pulse")
            
        """
        if data2save is None:
            data2save=zdata
        else:
            if data2save.any() is None:
                data2save=zdata

        if plot is not None and plot:
            pcm=plt.pcolormesh(xvar, yvar, zdata)
            plt.xlabel(xvarname)
            plt.ylabel(yvarname)
            plt.colorbar(pcm, label=zdataname)
            plt.title(title)
            
        if save is not None and save:
                today=date.today()
                if trial is None:
                    trial="01"
                if plot is not None and plot:
                    if os.path.exists('../images/{}'.format(q_name)) == False:
                        os.mkdir('../images/{}'.format(q_name))
                    plt.savefig('../images/{}/Stark-{}-{}-{}.pdf'.format(q_name, title, today, trial))
                
                if os.path.exists('../measurement_data/{}'.format(q_name)) == False:
                        os.mkdir('../measurement_data/{}'.format(q_name))
                
                df=pd.DataFrame(data2save, index=yvar, columns=xvar)
                df.to_csv('../measurement_data/{}/Data-Stark-{}-{}-{}.csv'.format(q_name, title, today, trial), mode='a', header=True)
                
                datapath = "../measurement_data/{}/Settings-Stark-{}-{}-{}.pickle".format(q_name, title, today, trial)
                for key, value in kwargs.items():
                    config[key] = value
                dumptopickle(datapath, config)

#Create histogram
def hist(data=None, plot=True, ran=1.0, numbins=100):
    
    ig = data[0]
    qg = data[1]
    ie = data[2]
    qe = data[3]

    # numbins = 200
    
    xg, yg = np.median(ig), np.median(qg)
    xe, ye = np.median(ie), np.median(qe)

    if plot==True:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16,4))
        fig.tight_layout()

        axs[0].scatter(ig, qg, label='g', color='b', marker='.')
        axs[0].scatter(ie, qe, label='e', color='r', marker='.')
        axs[0].scatter(xg, yg, color='k', marker='o')
        axs[0].scatter(xe, ye, color='k', marker='o')
        axs[0].set_xlabel('I (a.u.)')
        axs[0].set_ylabel('Q (a.u.)')
        axs[0].legend(loc='upper right')
        axs[0].set_title('Unrotated')
        axs[0].axis('equal')
    """Compute the rotation angle"""
    theta = -np.arctan2((ye-yg),(xe-xg))
    """Rotate the IQ data"""
    ig_new = ig*np.cos(theta) - qg*np.sin(theta)
    qg_new = ig*np.sin(theta) + qg*np.cos(theta) 
    ie_new = ie*np.cos(theta) - qe*np.sin(theta)
    qe_new = ie*np.sin(theta) + qe*np.cos(theta)
    
    """New means of each blob"""
    xg, yg = np.median(ig_new), np.median(qg_new)
    xe, ye = np.median(ie_new), np.median(qe_new)
    
    xlims = [xg-ran, xg+ran]
    ylims = [yg-ran, yg+ran]

    if plot==True:
        axs[1].scatter(ig_new, qg_new, label='g', color='b', marker='.')
        axs[1].scatter(ie_new, qe_new, label='e', color='r', marker='.')
        axs[1].scatter(xg, yg, color='k', marker='o')
        axs[1].scatter(xe, ye, color='k', marker='o')    
        axs[1].set_xlabel('I (a.u.)')
        axs[1].legend(loc='lower right')
        axs[1].set_title('Rotated')
        axs[1].axis('equal')

        """X and Y ranges for histogram"""
        
        ng, binsg, pg = axs[2].hist(ig_new, bins=numbins, range = xlims, color='b', label='g', alpha=0.5)
        ne, binse, pe = axs[2].hist(ie_new, bins=numbins, range = xlims, color='r', label='e', alpha=0.5)
        axs[2].set_xlabel('I(a.u.)')       
        
    else:        
        ng, binsg = np.histogram(ig_new, bins=numbins, range = xlims)
        ne, binse = np.histogram(ie_new, bins=numbins, range = xlims)

    """Compute the fidelity using overlap of the histograms"""
    contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5*ng.sum() + 0.5*ne.sum())))
    tind=contrast.argmax()
    threshold=binsg[tind]
    fid = contrast[tind]
    if plot:
        axs[2].set_title(f"Fidelity = {fid*100:.2f}%")

    return fid, threshold, theta

#Labrad's function to maximize IQ visibility
def maximize_visibility(I0s,Q0s,I1s,Q1s,initial_center,plot=True):
    """Find the center of the pi pulse IQ scatter data to maximize visibility."""
    I0 = np.mean(I0s)
    Q0 = np.mean(Q0s)
    def visibility(p,plot=False):
        I0s_shift = I0s - (p[0]+I0)/2.0
        Q0s_shift = Q0s - (p[1]+Q0)/2.0
        I1s_shift = I1s - (p[0]+I0)/2.0
        Q1s_shift = Q1s - (p[1]+Q0)/2.0
        angle = np.angle(-1j*p[1]-p[0]+1j*Q0+I0)
        rot_mat = np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
        IQ0s = np.dot(rot_mat,np.vstack([I0s_shift,Q0s_shift]))
        IQ1s = np.dot(rot_mat,np.vstack([I1s_shift,Q1s_shift]))
        I0_rot = IQ0s[0]
        I1_rot = IQ1s[0]
        prob0 = float(I0_rot[I0_rot<0].size)/I0_rot.size
        prob1 = float(I1_rot[I1_rot<0].size)/I1_rot.size
        if plot is True:
            y0,x0 = np.histogram(I0_rot,bins=50,normed=True)
            y1,x1 = np.histogram(I1_rot,bins=50,normed=True)
            x0 = (x0[:-1]+x0[1:])/2.0
            x1 = (x1[:-1]+x1[1:])/2.0
            probs0 = np.cumsum(y0)*(x0[1]-x0[0])
            probs1 = np.cumsum(y1)*(x1[1]-x1[0])
            x_visibility = np.linspace(min([x0[0],x1[0]]),max([x0[-1],x1[-1]]),1001)
            vis = []
            for x in x_visibility:
                prob0_fit = np.interp(x,x0,probs0,left=0,right=1)
                prob1_fit = np.interp(x,x1,probs1,left=0,right=1)
                vis.append(prob1_fit - prob0_fit)
            best_vis = np.max(vis)

            fig,ax1 = plt.subplots()
            plt.sca(ax1)
            plt.hist(I0_rot,color='b',bins=50,alpha=0.5,label='|0>')
            plt.hist(I1_rot,color='r',bins=50,alpha=0.5,label='|1>')
            plt.xlabel('Projection Position')
            plt.ylabel('Probability Density')
            plt.legend(loc=2)
            ax2 = ax1.twinx()
            plt.sca(ax2)
            plt.plot(x0,probs0,'b',label='|0>')
            plt.plot(x1,probs1,'r',label='|1>')
            plt.plot(x_visibility,vis,'g',label='|1>-|0>')
            plt.ylabel('Probability')
            plt.ylim([0,1])
            plt.legend(loc=6)
            plt.grid(True,which='both')
            plt.title('Best Visibility %.3f'%best_vis)
            plt.show()
        return prob1 - prob0

    def fit_func(p):
        return -visibility(p,plot=False)
    res = minimize(fit_func,initial_center,method='Nelder-Mead')
    if plot:
        visibility(res.x,plot=True)
    return -res.fun

#Fit functions
def sinfunc(x,A,W,P,L):
    return np.sin(W*x+P)*A+L

def exp_decay(x, A, B, T):
    return A*np.exp(-x/T)+B

def hahn_decay(x, A, T): #Hahn decay forces passage through origin.
    return A-A*np.exp(-x/T)

def p_decay(x, A, P, B):
    return A*np.power(P, x)+B

def t2fit(x, A, W, T, B, C):
    return np.exp(-x/T)*np.sin(W*x+C)*A+B
#Find nearest
def find_nearest(array,value):
    idx = np.searchsorted(array[:,0], value, side="left")
    if idx > 0 and (idx == len(array[:,0]) or math.fabs(value - array[idx-1,0]) < math.fabs(value - array[idx,0])):
        return array[idx-1,1]
    else:
        return array[idx,1]

def g_check(g_tuple, f_tuple, use_dbm):
    """Checks that gain parameters are within board limits. Converts gain units to dBm if use_dbm==True
    :param g_tuple: Tuple of three values, (g_start, g_stop, g_num). Sets gain values for sweep in linspace style.
    :param f_tuple: Tuple of three values, (f_start, f_stop, f_num).  Sets freq values for sweep in linspace style.
    :param use_dbm: Bool. If True, take g_tuple values to be dBm and convert them to gain using dbm2gain(), if False do nothing.
    :param nqz: Int. Nyquist zone used by DAC, value is passed to dbm2gain function.
    :param balun: Int. Balun used by DAC, value is passed to dbm2gain function.
    :returns: Error messages and code stops.g
    """
    #LLPS, LLKV, LLYD, LLLL
    f_start=f_tuple[0]; f_stop=f_tuple[1]
    g_start=g_tuple[0]; g_stop=g_tuple[1]
    if use_dbm:
        g_start=int(dbm2gain(g_start, (f_start+f_stop)/2, 2, 3))
        g_stop=int(dbm2gain(g_stop, (f_start+f_stop)/2, 2, 3))
    if g_stop > 32767:
        raise Exception("Gain exceeds board specifications. DAC output is limited to +32767 DAC units")
    if g_start < 0:
        temp=input("Warning: Negative Gain! While QICK specifications say DAC is 15 bit signed, the sign only determines the phase not power. So minimum power occurs at 0, not -32767 DAC units. Enter anything to continue ")
    if g_start < -32767:
        raise Exception("Gain exceeds board specifications. DAC output is limited to -32767 DAC units")

def dc2gain(gen, b_volt):
    path = "./constants/dcgain.npz"
    convdict = np.load(path, allow_pickle=True)
    gain2dcarr = convdict['gain2dc']
    dc_offset = convdict['dc_offset']
    gain = (b_volt-dc_offset[gen])/gain2dcarr[gen]
    return(gain)

def gain2dc(gen, gain):
    path = "./constants/dcgain.npz"
    convdict = np.load(path, allow_pickle=True)
    gain2dcarr = convdict['gain2dc']
    dc_offset = convdict['dc_offset']
    b_volt = dc_offset[gen] + gain2dcarr[gen]*gain
    return b_volt

def baseround(value, base=16):
    return int(base * round(value/base))

def gen_dc_waveform(gen, b_volt, m_volt=None, freq=None):
    
    path="./constants/dcgain.npz"
    convdict = np.load(path, allow_pickle=True)
    if freq:
        period=1/(freq*0.0001453218005952381)
        if period>65536:
            raise Exception('Modulating frequency needs to be higher')
        if period<16:
            raise Exception('Modulating frequency needs to be lower')
        period=period//16*16
        newfreq=1/(period*0.0001453218005952381)
        print("Frequency set to {}".format(newfreq))
        x_points=np.arange(period)
        y_data=m_volt/convdict['gain2dc'][gen]*np.sin(2*np.pi/period*x_points)
    
    else:
        y_data=np.zeros(48)

    y_data+=(b_volt-convdict["dc_offset"][gen])/convdict["gain2dc"][gen]

    return y_data
"""
def gen_dc_waveform2(gen, b_volt, dc_sequence, offset=0, m_volt=None, freq=None):
    
    Helper function to generate a dc waveform given a basis voltage, and
    optionally a modulating voltage and its frequency
    
    Voltage range is [-0.7303333268, 0.6903873267999999] V
    
    :param gen: the DC output DAC
    :param b_volt: basis voltage [V]
    :param dc_sequence: array of tuples. Each tuple is the (amplitude, length) of the DC flattop
    
    :param m_volt: modulating voltage [V]
    :param freq: frequency of modulating voltage [MHz]

    dict_cfg = loadfrompickle("Q_B")
    if freq is not None:
        period=1/freq/0.0026
        if period>65535:
            raise Exception('Modulating frequency needs to be higher')
        its=65535//period
        xlen=its*period
        xlen=xlen//16*16
        period=xlen/its
        x_points=np.arange(xlen)
        y_data=m_volt/dict_cfg["gain2dc"][gen]*np.sin(2*np.pi/period*x_points)
    
    else:
        if cycle_length is not None:
            y_data=np.zeros(cycle_length)
        else:
        y_data=np.zeros(65536)

   # if shape==None:
    y_data+=(b_volt-dict_cfg["dc_offset"][gen])/dict_cfg["gain2dc"][gen]
    elif (shape=="flat top" or shape=="flat_top") and (cycle_length is not None):
        y_data = 2*np.sin(np.linspace(0,(np.pi/2),cycle_length))
        y_data[int(np.floor(cycle_length/4)):int(np.ceil(cycle_length*3/4))] = 1
        y_data = y_data*((b_volt-dict_cfg["dc_offset"][gen])/dict_cfg["gain2dc"][gen])


    return y_data
"""
def gain2dbm(gain, freq, nqz, balun):
    dbm_list=np.load('./constants/dbm.npy')[balun*2+nqz-1]
    f=interpolate.interp1d(np.linspace(1e8,8e9,80), dbm_list)
    try:
        dbm=np.ones(len(gain))*f(freq*1e6)
    except:
        dbm=f(freq*1e6)
    dbm+=20*np.log10(gain/30000)
    return dbm

def dbm2gain(dbm, freq, nqz, balun):
    dbm_list=np.load('./constants/dbm.npy')[balun*2+nqz-1]
    f=interpolate.interp1d(np.linspace(1e8,8e9,80), dbm_list)
    dbm3e5=f(freq*1e6)
    gain=10**((dbm-dbm3e5)/20)*30000
    return gain
"""
def dc_gauss(mu=0, si=25, length=100, dc_bias=None, dc_pulse=None, dc_ch=None):

    maxv=int(dc2gain(dc_ch, dc_bias))
    pulsev=int(dc2gain(dc_ch, dc_pulse))-maxv
    x = np.arange(0, np.round(length)*16)
    y = maxv + (np.exp(-(x-mu)**2/(si*16)**2))*pulsev
    return y
"""
def flat_top(ramp_length=5, length=100, dc_bias=None, dc_ch=None):

    if np.abs(dc_bias)<3:
        maxv=int(dc2gain(dc_ch, dc_bias))
    else:
        maxv=dc_bias
    ramp=np.arange(ramp_length*16+1)
    gauss = np.exp(-(ramp-(ramp_length*8+1))**2/(ramp_length*4)**2)*maxv
    return np.concatenate((gauss[:ramp_length*8+1], np.ones(length*16)*maxv, gauss[ramp_length*8+1:]))

def iq_list_grapher(arr):
    dims = arr.shape
    if len(dims)==3 and dims[0]==1:
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Clock ticks")
        ax1.set_ylabel("a.u.")
        ax1.set_title("Single Pulse Sequence")
        for iq in arr:
            ax1.plot(iq[0], label="I value")
            ax1.plot(iq[1], label="Q value")
            ax1.plot(np.abs(iq[0]+1j*iq[1]), label="mag")
        ax1.legend()

    elif len(dims)==4 and dims[0]==1:
        arr_temp = arr[0]
        fig, ax2 = plt.subplots()
        counter = 0
        ax2.set_xlabel("Clock ticks")
        ax2.set_ylabel("a.u.")
        ax2.set_title("Single Pulse Sequence")
        for iq in arr_temp:
            ax2.plot(np.abs(iq[0]+1j*iq[1]), label="Pulse "+str(counter) + " Mag")
            counter = counter+1
        ax2.legend()
    else:
        raise Exception("Unrecognized iq_list shape. Please check you're specifying the iq_list from decimated readout.")

def labrad1d(q_name=None, title=None, path=r"V:/shared/data_vault_server_file/Jeffrey_Q_B.dir/", config=None, xdict=None, **kwargs):
    """
    Export QICK board data to labrad readable format by generating associated .csv and .ini file
    
    :param q_name: name of qubit used (passed into .ini file parameters) (string)
    :param title: title of the plot and of the save files (string)
    :param trial: number of trial, used in title and filenames. Default 1 (int)
    :param path: path relative to V:/shared/data_vault_server_file/. Default "Jeffrey_Q_B.dir/" (str)
    :param config: experiment configuration (dictionary)
    :param xdict: dictionary with throw-away key and tuple of x data (xdata, xname, xunits). (single entry dictionary w/ tuple)
    :param **kwargs: dictionary with throw-away keys and tuples of y data in the form (data, label, units, category) (dictionary w/ tuples)
    """
    
    #Basic checks
    if q_name == None or title == None or config == None or xdict == None:
        raise Exception("You must specify q_name, title, config, xdict, and the ydict in the **kwargs argument.")

    file_location = path

    #Checking file preamble number
    existing_files = [f for f in os.listdir(file_location) if f.endswith('.csv')]
    if existing_files:
        existing_numbers = [int(f.split(' - ')[0]) for f in existing_files]
        next_number = max(existing_numbers) + 1
    else:
        next_number = 0

    #Creating file name based off ^
    file_name = f"{next_number:05d} - %v{q_name}%g - {title}"

    #X variable stuff
    for temp, xtuple in xdict.items():
        xvar = xtuple[0]
        xvarname = xtuple[1]
        xvarunits = xtuple[2]
    
    #CSV creation
    save_data = xvar
    for temp, ytuple in kwargs.items():
        save_data = np.vstack((save_data, ytuple[0]))
    save_data = np.transpose(save_data)    
    csv_file_path = file_location + file_name + ".csv"
    np.savetxt(csv_file_path, save_data, delimiter=',')
    
    #.ini file creation
    file_path = file_location + file_name + ".txt" #write as .txt first+ then switch to .ini to avoid using a parser lol
    ini_file_path = file_location + file_name + ".ini"
    
    if os.path.exists(ini_file_path) == True:
        print(".ini file of the same name found! Deleting and overwriting.")
        os.remove(ini_file_path)
    
    with open(file_path, 'w') as file:
        file.write("[General] \n")
        date_str = datetime.now().date().strftime('%Y-%m-%d')
        time_str = datetime.now().time().strftime('%H:%M:%S')
        file.write("created = " + date_str + ", " + time_str + "\n")
        file.write("accessed = "+ date_str + ", " + time_str + "\n")
        file.write("modified = "+ date_str + ", "+ time_str+ "\n")
        file.write("title = "+ title + "\n")
        file.write("independent = 1"+ "\n") #only one because 1D
        file.write("dependent = " + str(len(kwargs)) + "\n") #only one because the way plotsave1D is set up. we can extend in future and remove this function
        file.write("parameters = "+ str(len(config))+ "\n")
        file.write("comments = 0"+ "\n")
        
        file.write("\n")
        
        file.write("[Independent 1]"+ "\n")
        file.write("label = "+ xvarname+ "\n")
        file.write("units = "+ xvarunits+ "\n") 
        
        file.write("\n")
        
        counter = 1
        for temp, ytuple in kwargs.items():
            file.write("[Dependent " + str(counter) + "]" + "\n")
            file.write("label = "+ ytuple[1]+ "\n")
            file.write("units = "+ ytuple[2]+ "\n")
            file.write("category = "+ ytuple[3]+ "\n")

            file.write("\n")
            counter += 1
        
        file.write("[Parameter 1]\n")
        file.write("label = Qubit Used \n")
        file.write("data = "+ str(q_name)+ "\n")
        file.write("\n")
            
        counter = 2
        for key,value in config.items():
            # print(key, value)
            file.write("[Parameter "+ str(counter)+ "]"+ "\n")
            file.write("label = "+ str(key)+ "\n")
            file.write("data = "+ str(value)+ "\n")
            file.write("\n")
            counter+=1
            
        file.write("[Comments]")
        
    os.rename(file_path, ini_file_path)

def labrad2d(q_name=None, title=None, path=r"V:/shared/data_vault_server_file/Jeffrey_Q_B.dir/", config=None, xdict=None, ydict=None, **kwargs):
    """
    Export QICK board data to labrad readable format by generating associated .csv and .ini file
    
    :param q_name: name of qubit used (passed into .ini file parameters) (string)
    :param title: title of the plot and of the save files (string)
    :param trial: number of trial, used in title and filenames. Default 1 (int)
    :param path: path relative to V:/shared/data_vault_server_file/. Default "Jeffrey_Q_B.dir/" (str)
    :param config: experiment configuration (dictionary)
    :param xdict: dictionary with throw-away key and tuple of x data (xdata, xname, xunits). (single entry dictionary w/ tuple)
    :param ydict: dictionary with throw-away key and tuple of x data (xdata, xname, xunits). (single entry dictionary w/ tuple)
    :param **kwargs: dictionary with throw-away keys and tuples of y data in the form (data, label, units, category) (dictionary w/ tuples)
    """

    #Basic checks
    if q_name == None or title == None or config == None or xdict == None or ydict==None:
        raise Exception("You must specify q_name, title, config, xdict, ydict, and the zdict(s) in the **kwargs argument.")
    
    file_location = path

    #Checking file preamble number
    existing_files = [f for f in os.listdir(file_location) if f.endswith('.csv')]
    if existing_files:
        existing_numbers = [int(f.split(' - ')[0]) for f in existing_files]
        next_number = max(existing_numbers) + 1
    else:
        next_number = 0

    #Creating file name based off ^
    file_name = f"{next_number:05d} - %v{q_name}%g - {title}"
    
    #x, y var stuff
    for temp, xtuple in xdict.items():
        xvar = xtuple[0]
        xvarname = xtuple[1]
        xvarunits = xtuple[2]
    for temp, ytuple in ydict.items():
        ydata = ytuple[0]
        ydataname = ytuple[1]
        ydataunits = ytuple[2]
    
    new_xvar = np.repeat(xvar, len(ydata))
    new_ydata = np.tile(ydata, len(xvar))
    
    save_data = np.vstack((new_xvar, new_ydata))
    for temp, ztuple in kwargs.items():
        new_zdata = ztuple[0].flatten()
        save_data = np.vstack((save_data, new_zdata))

    save_data = np.transpose(save_data)
    csv_file_path = file_location + file_name + ".csv"
    np.savetxt(csv_file_path, save_data, delimiter=',')
    
    #ini file creation
    file_path = file_location + file_name + ".txt" #write as .txt first then switch to .ini to avoid using a parser lol
    ini_file_path = file_location + file_name + ".ini"
    
    if os.path.exists(ini_file_path) == True:
        print(".ini file of the same name found! Deleting and overwriting.")
        os.remove(ini_file_path)
    
    with open(file_path, 'w') as file:
        file.write("[General] \n")
        date_str = datetime.now().date().strftime('%Y-%m-%d')
        time_str = datetime.now().time().strftime('%H:%M:%S')
        file.write("created = " + date_str + ", " + time_str + "\n")
        file.write("accessed = "+ date_str + ", " + time_str + "\n")
        file.write("modified = "+ date_str+ ", "+ time_str+ "\n")
        file.write("title = "+ title + "\n")
        file.write("independent = 2" + "\n") #only one because 1D
        file.write("dependent = " + str(len(kwargs)) + "\n") #only one because the way plotsave1D is set up. we can extend in future and remove this function
        file.write("parameters = " + str(len(config)) + "\n")
        file.write("comments = 0" + "\n")
        
        file.write("\n")
        
        file.write("[Independent 1]"+ "\n")
        file.write("label = "+ xvarname+ "\n")
        file.write("units = "+ xvarunits+ "\n") 
        
        file.write("\n")
        
        file.write("[Independent 2]"+ "\n")
        file.write("label = "+ ydataname + "\n")
        file.write("units = "+ ydataunits+ "\n") 
        
        file.write("\n")
        
        counter = 1
        for temp, ztuple in kwargs.items():
            file.write("[Dependent " + str(counter) + "]\n")
            file.write("label = "+ ztuple[1]+ "\n")
            file.write("units = "+ ztuple[2]+ "\n")
            file.write("category = "+ ztuple[3] + "\n")

            file.write("\n")
            counter += 1
        
        file.write("[Parameter 1]\n")
        file.write("label = Qubit Used \n")
        file.write("data = "+ str(q_name)+ "\n")
        file.write("\n")
            
        counter = 2
        for key,value in config.items():
            file.write("[Parameter "+ str(counter)+ "]"+ "\n")
            file.write("label = "+ str(key)+ "\n")
            file.write("data = "+ str(value)+ "\n")
            file.write("\n")
            counter+=1
            
        file.write("[Comments]")
        
    os.rename(file_path, ini_file_path)

    # def initialize_labradsave(path, name, independent, dependent, param_dict, qubit='Q1'):
    #     """Makes csv and ini file for labrad saving.
    #     :param path: String. Datavault path to save to.
    #     :param name: String. Measurement name.
    #     :param independent: Array. Independent variable(s). If one independent variable is specified, save data in 1D format. If two are specified save in 2D format.
    #     :param dependent: Array. Dependent variable(s).
    #     :param param_dict: Dictionary. Dictionary of experiment parameters to be saved to .ini file.
    #     :param qubit: String. Qubit ID for file name.
    #     """

    # def update_labradsave():
    #     """Adds single datapoint to labrad csv file.
    #     :param path: String. Datavault path to save to.
    #     :param name: String. Measurement name.
    #     :param independent: Array. Independent variable(s). If one independent variable is specified, save data in 1D format. If two are specified save in 2D format.
    #     :param dependent: Array. Dependent variable(s).
    #     :param param_dict: Dictionary. Dictionary of experiment parameters to be saved to .ini file.
    #     :param qubit: String. Qubit ID for file name.
    #     """
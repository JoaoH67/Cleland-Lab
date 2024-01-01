#Pyro 4 Preamble
#For now, this cell must be run in this python file.
#Will be coming up with a fix later
import Pyro4
from qick import QickConfig
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION=4

# ip = input("Please enter IP address")
ns_host = "192.168.0.107" #IP Address of QICK board
ns_port = 8888 #Change depending on which board you're connecting to
proxy_name = "Kentrell" #Change depending on how you initialized NS

ns = Pyro4.locateNS(host=ns_host, port=ns_port)
soc = Pyro4.Proxy(ns.lookup(proxy_name))
soc.get_cfg()
soccfg = QickConfig(soc.get_cfg())
print("Ignore error that says Could not import QickSoc: No Module named 'pynq'. That will not impact the code.")
print("...")

#Actual premable
import copy
from qick import *
from qick.averager_program import *
from qick.helpers import gauss
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import leastsq,minimize
from scipy import stats
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from helpers import *

#Global default parameter location
global default_param_path, default_save_path
default_param_path = r"C:/C:\Users\Tarly\Desktop\J+J_QICK\measurement_data\parameters.pickle"
default_save_path = r"V:/shared/data_vault_server_file/Jeffrey_Q_B.dir/"

class LoopbackProgram(AveragerProgram):
    def initialize(self):
        cfg=self.cfg   

        self.declare_gen(ch=cfg["pulse_ch"], nqz=cfg["nqz"]) #nqz sets the Nyquist zone. Preferrably, use nqz=1 for pulses under 3440MHz

        global freq
        freq = self.freq2reg(cfg["pulse_freq"],gen_ch=cfg["pulse_ch"]) #converts frequency to register frequency
        self.set_pulse_registers(ch=cfg["pulse_ch"], style="const", length=cfg["length"], freq=freq, phase=0, gain=cfg["pulse_gain"]) #sets the pulse to be played
        self.synci(200) #small delay to synchronize everything
    
    def body(self):
        
        self.pulse(ch=self.cfg["pulse_ch"]) #sends the pulse
        self.sync_all(10)
        self.wait_all(10) #waits a specified number of clock ticks. Here, it's none.

def ContinuousSignal(ch=None, nqz=None, freq=None, gain=None, use_dbm=False, time=100):
    """Function to send a continuous signal of set frequency and gain. 
    Autochecks nqz. 
    2.5 ns delay every 23.25 us as clock cycles over
    
    :param ch: output channel [int]
    :param nqz: Nyquist zone [int]
    :param freq: frequency of the signal [MHz]
    :param gain: gain of the signal [a.u.]
    :param use_dbm: use dbm units for gain? [dBm]
    :param time: time of pulse in us
    """ 

    if ch==None or nqz==None or freq==None or gain==None:
        raise Exception("You must specify the channel, Nyquist zone, and frequency.")

    if use_dbm==True:
        gain=dbm2gain(gain, freq, nqz, 3)

    if freq > 3440 and nqz==1:
        print("The Nyquist frequency is 3.44 GHz. You have chosen a frequency in the 2nd NQZ but set NQZ=1.")
    
    loops = int(time/soccfg.cycles2us(10000))

    config={"pulse_ch":ch,
            "reps":loops,
            "relax_delay":1.0,
            "length":10000,
            "pulse_gain":int(gain),
            "pulse_freq":freq,
            "soft_avgs":1,
            "nqz":nqz
           }

    prog =LoopbackProgram(soccfg, config)

    prog.acquire(soc, load_pulses=True, progress=True, debug=False)

class DCLoopbackProgram(AveragerProgram):
    def initialize(self):
        cfg=self.cfg
        res_ch = cfg["dc_ch"] #defines the output
        
        self.add_pulse(ch=cfg["dc_ch"], name="dc", idata=gen_dc_waveform(cfg["dc_ch"], cfg["voltage"]))
        
        self.set_pulse_registers(ch=cfg["dc_ch"], outsel="input", style="arb", phase=0, 
                                freq=0, waveform="dc", gain=32767, mode="periodic")
        self.synci(200) #small delay to synchronize everything
    
    def body(self):
        
        self.pulse(ch=self.cfg["dc_ch"]) #sends the pulse
        self.wait_all() #waits a specified number of clock ticks. Here, it's none.

def DCSignalCont(ch=None, voltage=None):
    """Function to send a continuous signal of set frequency and gain
    
    :param ch: output channel [int]
    :param voltage: voltage [V]
    
    """ 

    if ch==None or voltage==None:
        raise Exception("You must specify the channel and voltage.")
    
    config={"dc_ch":ch,
            "reps":1,
            "relax_delay":1.0,
            "length":1000,
            "voltage":voltage,
            "soft_avgs":1
           }

    prog = DCLoopbackProgram(soccfg, config)

    avgq, avgi = prog.acquire(soc, load_pulses=True, progress=True, debug=False)

class RRProgram(RAveragerProgram):
    #Unused as of yet. Need to check if it's necessary to run TWPA with board. 
    def initialize(self):
        cfg=self.cfg 
        self.declare_gen(ch=cfg["res_ch"], nqz=2)
        
        
        self.r_rp=self.ch_page(self.cfg["res_ch"])
        self.r_gain=self.sreg(cfg["res_ch"], "gain") #Declare the address where the value of gain will be registered for the sweep
        
        #Declare readout channels
        for ch in [0]:
            self.declare_readout(ch=ch, length=cfg["readout_length"],
                                 freq=cfg["frequency"], gen_ch=cfg["res_ch"])
            
        freq = self.freq2reg(cfg["frequency"], gen_ch=cfg["res_ch"], ro_ch=0)
        
        #Creates the waveform for the readout pulse
        self.add_gauss(ch=cfg["res_ch"], name="measure", sigma=cfg["res_length"]/4, length=cfg["res_length"])
        self.set_pulse_registers(ch=cfg["res_ch"], style="flat_top", waveform="measure", freq=freq, length=cfg["res_length"], 
                                 phase=0, gain=cfg["start"])

        #DC pulse that runs for the duration
        waveform = np.ones((cfg["res_length"]+2*cfg["bias_settling"])*16)*int(dc2gain(cfg["dc_ch"],cfg["voltage"]))
        self.add_pulse(ch=cfg["dc_ch"], name="dc", idata=waveform)
        self.set_pulse_registers(ch=cfg["dc_ch"], outsel='input', style='arb', phase=0, 
                                 freq=0, waveform='dc', gain=32767, mode='oneshot')

        #check if TWPA is needed, set pulse registers if needed.
        if cfg["twpa"]: 
            self.declare_gen(ch=cfg["twpa_ch"], nqz=2)
            twpa_freq = self.freq2reg(cfg["twpa_freq"], gen_ch=cfg["twpa_ch"], ro_ch=0)
            self.set_pulse_registers(ch=cfg["twpa_ch"], style="const", length=cfg["res_length"], freq=twpa_freq, phase=0, gain=cfg["twpa_gain"])
    
        self.synci(200)
    
    def body(self):
        cfg=self.cfg
        
        self.pulse(ch=cfg["dc_ch"], t=0)
        
        if cfg["twpa"]:
            self.pulse(ch=cfg["twpa_ch"], t=cfg["bias_settling"])

        #Triggers waveform acquisition
        self.trigger(adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=cfg["adc_trig_offset"], 
                     t=cfg["bias_settling"])
        
        self.pulse(ch=cfg["res_ch"], t=cfg["bias_settling"])
        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))
        
    def update(self):
        self.mathi(self.r_rp, self.r_gain, self.r_gain, '+', self.cfg["step"])

class DCRRProgram(RAveragerProgram):
    def initialize(self):
        cfg=self.cfg 
        self.declare_gen(ch=cfg["res_ch"], nqz=2)
        
        self.r_rp=self.ch_page(self.cfg["dc_ch"])
        self.r_gain=self.sreg(cfg["dc_ch"], "gain")
        
        #Declare readout channels
        for ch in [0]:
            self.declare_readout(ch=ch, length=cfg["readout_length"],
                                 freq=cfg["frequency"], gen_ch=cfg["res_ch"])
            
        freq = self.freq2reg(cfg["frequency"], gen_ch=cfg["res_ch"], ro_ch=0)
        
        #Creates the waveform for the readout pulse
        self.add_gauss(ch=cfg["res_ch"], name="measure", sigma=cfg["res_length"]/4, length=cfg["res_length"])
            
        self.set_pulse_registers(ch=cfg["res_ch"], style="flat_top", waveform="measure", freq=freq, length=cfg["res_length"], 
                                 phase=0, gain=cfg["pulse_gain"])
        
        rolen = (cfg["res_length"])*16
        waveform = np.ones(rolen)*cfg["start"]
        self.add_pulse(ch=cfg["dc_ch"], name="dc", idata=waveform)
        self.set_pulse_registers(ch=cfg["dc_ch"], outsel='input', style='arb', phase=0, 
                                 freq=0, waveform='dc', gain=32766, mode='periodic')

        self.synci(200)
    
    def body(self):
        cfg=self.cfg
            
        #Triggers waveform acquisition
        self.trigger(adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=cfg["adc_trig_offset"])
        
        self.pulse(ch=cfg["dc_ch"])
        self.pulse(ch=cfg["res_ch"])
        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))
        
    def update(self):
        self.mathi(self.r_rp, self.r_gain, self.r_gain, '+', self.cfg["step"])

def FreqRR(q_name=None, pkl_path=default_param_path, ro_length=None, f_tuple=None, gain=1000, use_dbm=False, s21=False, dc=0.0, twpa=False, reps=300,
save_path=default_save_path, title=None, collect=None):
#### NOT FUNCTIONAL ####
    
    """
    2D scan frequency and gain to find readout resonator

    :param q_name: name of the qubit in dict_cfg
    :param ro_length: length of the probe pulse [us]
    :param f_tuple: (start freq [MHz], stop freq [MHz], number of samples [int])
    :param g_tuple: (start gain [DAC units/dBm], stop gain [DAC units/dBm], number of samples [int])
    :param use_dbm: if True, g_tuple entries are in dBm. if False, enter g_tuple entires are in DAC units.
    :param s21: plot/save s21 instead of uncorrected amplitude
    :param dc: bias voltage [V]
    :param title: title of plot and save file (and .ini file for labrad grapher)
    :param path: path to save labrad grapher files into. Defaults to "Jeffrey_Q_B.dir/"
    :param collect: return the raw amplitude/s21 and angle data.
    """

    if q_name==None or ro_length==None or f_tuple==None:
        raise Exception("You must specify q_name, ro_length, f_start, f_stop, f_expts. The default gain is 1000. To use dbm units for power, specify the gain measurements in units of dbm and set use_dbm=True")
   
    
    #setting up sweep parameters
    f_start=f_tuple[0]; f_stop=f_tuple[1]; f_expts=f_tuple[2]

    units = "a.u."
    if use_dbm == True:
        units = "dBm"
        gain=int(dbm2gain(gain, (f_start+f_stop)/2, 2, 3))

    expt_cfg={"reps":reps, "relax_delay":10, "f_start":f_start, "f_stop":f_stop, "f_expts":f_expts, 
              "pulse_gain":gain, "voltage":dc, "twpa":twpa}

    dict_cfg = loadfrompickle(q_name, path=pkl_path)
    config={**dict_cfg, **expt_cfg}

    if config["threshold"] is not None:
        corr=config["readout_length"]
    else:
        corr=1
    
    config["res_length"]=soccfg.us2cycles(ro_length,gen_ch=2) #converts length to clock cycles
    f_range=np.linspace(config["f_start"], config["f_stop"], config["f_expts"])

    results=[]
    s21_amps=[]
    for i in tqdm(f_range):
        config["frequency"]=i,
        rspec=RR1DProgram(soccfg, config)
        avgi,avgq=rspec.acquire(soc, load_pulses=True, progress=False) #calls the previous cell with the actual pulse sequences and measurements
        results.append((avgi[0][0]+1j*avgq[0][0])*corr)
        s21_amp = 20*np.log10(np.abs((avgi[0][0]+1j*avgq[0][0])*corr) / np.array(expt_pts))
        s21_amps.append(s21_amp)

    #All relevant data
    amps=np.abs(results)
    angles=np.angle(results)
    s21_amps = np.array(s21_amps)
    
    if use_dbm==True:
        yvar=gain2dbm(expt_pts, (f_start+f_stop)/2, 2, 3)
        yvarname="Power"
    else:
        yvar=expt_pts
        yvarname="DAC Gain"

    if s21:
        if title==None:
            title = "RR Spectroscopy, S21"

        #setting up and saving using labrad format
        f_dict = {'frange':(f_range, "Frequency", "MHz")} 
        s21_tuple = (s21_amps, "S21 Amplitude", "dBm", "S21 Amplitude") ; angles_tuple = (angles, "Angle", "Radians", "Angle")
        labrad1d(q_name=q_name, path=save_path, title=title, config=dict_cfg, xdict=f_dict, s21=s21_tuple, angles=angles_tuple)

        if collect:
            return s21_amps.T, angles.T

    else:
        if title==None:
            title = "RR Spectroscopy"

        #setting up and saving using labrad format
        f_dict = {'frange':(f_range, "Frequency", "MHz")}
        amps_tuple = (amps, "Amplitude", "a.u.", "Category 1") ; angles_tuple = (angles, "Angle", "Radians", "Angle")
        labrad2d(q_name=q_name, path=save_path, title=title, config=dict_cfg, xdict=f_dict, ydict=g_dict, amps=amps_tuple, angles=angles_tuple)

        if collect:
            return amps.T, angles.T

def FreqGainRR(q_name=None, pkl_path=default_param_path, ro_length=None, f_tuple=None, g_tuple=None, use_dbm=None, s21=False, dc=0.0, twpa=False, reps=300,
save_path=default_save_path, title=None, collect=None):
    
    """
    2D scan frequency and gain to find readout resonator

    :param q_name: name of the qubit in dict_cfg
    :param ro_length: length of the probe pulse [us]
    :param f_tuple: (start freq [MHz], stop freq [MHz], number of samples [int])
    :param g_tuple: (start gain [DAC units/dBm], stop gain [DAC units/dBm], number of samples [int])
    :param use_dbm: if True, g_tuple entries are in dBm. if False, enter g_tuple entires are in DAC units.
    :param s21: plot/save s21 instead of uncorrected amplitude
    :param dc: bias voltage [V]
    :param title: title of plot and save file (and .ini file for labrad grapher)
    :param path: path to save labrad grapher files into. Defaults to "Jeffrey_Q_B.dir/"
    :param collect: return the raw amplitude/s21 and angle data.
    """

    if q_name==None or ro_length==None or f_tuple==None or g_tuple==None:
        raise Exception("You must specify q_name, ro_length, f_start, f_stop, f_expts, g_start, g_stop, and g_expts. To use dbm units for power, specify the gain measurements in units of dbm and set use_dbm=True")
   
    g_check(g_tuple, f_tuple, use_dbm)

    #setting up sweep parameters
    f_start=f_tuple[0]; f_stop=f_tuple[1]; f_expts=f_tuple[2]
    g_start=g_tuple[0]; g_stop=g_tuple[1]; g_expts=g_tuple[2]

    units = "a.u."
    if use_dbm == True:
        units = "dBm"
        g_start=int(dbm2gain(g_start, (f_start+f_stop)/2, 2, 3))
        g_stop=int(dbm2gain(g_stop, (f_start+f_stop)/2, 2, 3))

    expt_cfg={"reps":reps, "relax_delay":10, "f_start":f_start, "f_stop":f_stop, "f_expts":f_expts, 
              "start":g_start, "step":int((g_stop-g_start)/g_expts), "expts":g_expts, "voltage":dc, "twpa":twpa}

    dict_cfg = loadfrompickle(q_name, path=pkl_path)
    config={**dict_cfg, **expt_cfg}

    if config["threshold"] is not None:
        corr=config["readout_length"]
    else:
        corr=1
    
    config["res_length"]=soccfg.us2cycles(ro_length,gen_ch=2) #converts length to clock cycles
    f_range=np.linspace(config["f_start"], config["f_stop"], config["f_expts"])

    results=[]
    s21_amps=[]
    for i in tqdm(f_range):
        config["frequency"]=i,
        rspec=RRProgram(soccfg, config)
        expt_pts, avgi,avgq=rspec.acquire(soc, load_pulses=True, progress=False) #calls the previous cell with the actual pulse sequences and measurements
        results.append((avgi[0][0]+1j*avgq[0][0])*corr)
        s21_amp = 20*np.log10(np.abs((avgi[0][0]+1j*avgq[0][0])*corr) / np.array(expt_pts))
        s21_amps.append(s21_amp)

    #All relevant data
    expt_pts=np.array(expt_pts)
    amps=np.abs(results)
    angles=np.angle(results)
    s21_amps = np.array(s21_amps)
    
    if use_dbm==True:
        yvar=gain2dbm(expt_pts, (f_start+f_stop)/2, 2, 3)
        yvarname="Power"
    else:
        yvar=expt_pts
        yvarname="DAC Gain"

    if s21:
        if title==None:
            title = "RR Spectroscopy, S21"

        #setting up and saving using labrad format
        f_dict = {'frange':(f_range, "Frequency", "MHz")} ; g_dict = {'frange':(yvar, yvarname, units)}
        s21_tuple = (s21_amps, "S21 Amplitude", "dB", "S21 Amplitude") ; angles_tuple = (angles, "Angle", "Radians", "Angle")
        labrad2d(q_name=q_name, path=save_path, title=title, config=dict_cfg, xdict=f_dict, ydict=g_dict, s21=s21_tuple, angles=angles_tuple)

        if collect:
            return s21_amps.T, angles.T

    else:
        if title==None:
            title = "RR Spectroscopy"

        #setting up and saving using labrad format
        f_dict = {'frange':(f_range, "Frequency", "MHz")} ; g_dict = {'frange':(yvar, yvarname, units)}
        amps_tuple = (amps, "Amplitude", "a.u.", "Category 1") ; angles_tuple = (angles, "Angle", "Radians", "Angle")
        labrad2d(q_name=q_name, path=save_path, title=title, config=dict_cfg, xdict=f_dict, ydict=g_dict, amps=amps_tuple, angles=angles_tuple)

        if collect:
            return amps.T, angles.T

def DcFreqRR(q_name=None, pkl_path=default_param_path, ro_length=None, reps=300,
f_tuple=None, stl_time=1, dc=None, dc_tuple=None, gain=None, use_dbm=None, s21=False, title=None, save_path=default_save_path, collect=None):
    
    """
    finds the frequency of the readout resonator
    
    :param q_name: name of the qubit in dict_cfg
    :param ro_length: length of the probe pulse [us]
    :param f_start: start frequency [MHz]
    :param f_stop: stop frequency [MHz]
    :param f_expts: # of frequency points (int)
    :param dc_start: start dc bias voltage [V]
    :param dc_stop: stop dc bias voltage [V]
    :param dc_expts: # of dc points (int)
    :param gain: pulse amplitude [a.u.]
    :param use_dbm: specify gain in dbm. Autoconvert to QICK units [Bool] 

    """

    if q_name==None or ro_length==None or f_tuple==None or gain==None:
        raise Exception("You must specify q_name, ro_length, f_start, f_stop, f_expts and gain. To use dbm units for power, specify the gain measurements in units of dbm and set use_dbm=True")

    f_start = f_tuple[0] ; f_stop = f_tuple[1] ; f_expts = f_tuple[2]

    if use_dbm==True:
        gain=int(dbm2gain(gain, (f_start+f_stop)/2, 2, 3))
    

    expt_cfg={"reps":reps, "relax_delay":10, "f_start":f_start, "f_stop":f_stop, "f_expts":f_expts, "pulse_gain":gain}
    
    dict_cfg = loadfrompickle(q_name, pkl_path)
    config={**dict_cfg, **expt_cfg}
    
    if dc==None:
        dc_start = dc_tuple[0] ; dc_stop = dc_tuple[1] ; dc_expts = dc_tuple[2]
        g_start=int(dc2gain(config["dc_ch"],dc_start))
        g_stop=int(dc2gain(config["dc_ch"], dc_stop))
        config["start"]=int(g_start)
        config["step"]=int((g_stop-g_start)/dc_expts)
        config["expts"]=dc_expts
        v_range=np.linspace(dc_start, dc_stop, dc_expts)
    else:
        config["voltage"]=dc
    
    config["res_length"]=soccfg.us2cycles(ro_length,gen_ch=2) #converts length to clock cycles
    config["stl_time"]=soccfg.us2cycles(stl_time, gen_ch=2) 

    if config["threshold"] is not None:
        corr=config["readout_length"]
    else:
        corr=1
    f_range=np.linspace(config["f_start"], config["f_stop"], config["f_expts"])

    if dc==None:
        results=[]
        for i in tqdm(f_range):
            config["frequency"]=i
            rspec=DCRRProgram(soccfg, config)
            expt_pts,avgi,avgq=rspec.acquire(soc, load_pulses=True, progress=False) #calls the previous cell with the actual pulse sequences and measurements
            results.append((avgi[0][0]+1j*avgq[0][0])*corr)
        
        zampl=np.abs(results)
        zangl=np.angle(results)

        s21_amps = 20*np.log10(zampl/gain)

        if title==None:
            if s21:
                title = "DC RR Spectroscopy 2D, S21"
            else:
                title = "DC RR Spectroscopy 2D"
        
        if s21:
            amplitudes_tuple = (s21_amps, "S21 Amplitude", "dBm", "S21 Amplitude")
        else:
            amplitudes_tuple = (zampl, "Amplitude", "DAC Units", "Amplitude")

        #plotting using labrad format
        f_dict = {'frange':(f_range, "Qubit Frequency", "MHz")} ; g_dict = {'grange':(v_range, "DC Voltage", "V")}
        angles_tuple = (zangl, "Angle", "Radians", "Angle")
        labrad2d(q_name=q_name, path=save_path, title=title, config=dict_cfg, xdict=f_dict, ydict=g_dict, amplitudes=amplitudes_tuple, angles=angles_tuple)

        if collect:
            return zampl, zangl

    else:
        results=[]
        for i in tqdm(f_range):
            config["frequency"]=i
            rspec=DCRRProgram1D(soccfg, config)
            avgi,avgq=rspec.acquire(soc, load_pulses=True, progress=False) #calls the previous cell with the actual pulse sequences and measurements
            results.append(np.abs(avgi[0][0]+1j*avgq[0][0])*corr)
        soc.reset_gens()
        
        zampl=np.array(results)
        zangl=np.angle(results)
        
        if title==None:
            title = "DC RR Spectroscopy 2D"
        
        #plotting using labrad format
        f_dict = {'frange':(f_range, "Qubit Frequency", "MHz")}
        amplitudes_tuple = (zampl, "Amplitude", "DAC Units", "Amplitude") ; angles_tuple = (zangl, "Angle", "Radians", "Angle") 
        labrad1d(q_name=q_name, path=save_path, title=title, config=dict_cfg, xdict=f_dict, amplitudes=amplitudes_tuple, angles=angles_tuple)
        
        if collect:
            return zampl, zangl
    soc.reset_gens()

class PulseProbeSpectroscopyProgram1D(AveragerProgram):
    def initialize(self):
        cfg=self.cfg
        
        self.q_rp=self.ch_page(cfg["qubit_ch"])     # get register page for qubit_ch
        self.r_delay = 3
        self.regwi(self.q_rp, self.r_delay, cfg["drive_ro_delay"])

        self.declare_gen(ch=cfg["qubit_ch"], nqz=2)
        
        for ch in [0]:
            self.declare_readout(ch=ch, length=cfg["readout_length"],
                                 freq=cfg["f_res"], gen_ch=cfg["qubit_ch"])
             
        global freq
        freq = self.freq2reg(cfg["f_res"], gen_ch=cfg["qubit_ch"], ro_ch=0)
        
        global phase
        phase=self.deg2reg(cfg["phase"])
        
        self.add_DRAG(ch=cfg["qubit_ch"], name="drive", length=cfg["probe_length"], sigma=cfg["probe_length"]/4,
                      delta=cfg["delta"], alpha=cfg["alpha"])
        self.add_gauss(ch=cfg["res_ch"], name="measure", sigma=cfg["res_sigma"]/4, length=cfg["res_sigma"])
        self.add_pulse(ch=cfg["dc_ch"], name="dc", idata=gen_dc_waveform(cfg["dc_ch"], cfg["voltage"]))
            
        self.set_pulse_registers(ch=cfg["dc_ch"], outsel="input", style="arb", phase=0, 
                                freq=0, waveform="dc", gain=32767, mode="periodic")
        self.set_pulse_registers(ch=cfg["qubit_ch"], style="arb", freq=self.freq2reg(cfg["qubit_freq"], gen_ch=cfg["qubit_ch"]), 
                                 gain=cfg["qubit_gain"], waveform="drive", phase=0)
        self.set_pulse_registers(ch=cfg["res_ch"], style="flat_top", waveform="measure", freq=freq, length=cfg["res_length"], 
                                 gain=cfg["res_gain"], phase=phase)
        self.sync_all(200)
    
    def body(self):
        cfg=self.cfg
        
        self.pulse(ch=cfg["dc_ch"])
           
        #drive pulse
        self.pulse(ch=self.cfg["qubit_ch"])
        
        self.sync(self.q_rp, self.r_delay)

        #measure pulse
        self.trigger(adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=cfg["adc_trig_offset"], t=int(cfg["probe_length"]+cfg["drive_ro_delay"]))
        
        
        self.pulse(ch=cfg["res_ch"], t=int(cfg["probe_length"]+cfg["drive_ro_delay"]))
        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))

def QubitSpectroscopy1D(q_name, pkl_path=default_param_path, probe_length=None, gain=None, use_dbm=False, reps=300, f_tuple=None, dc=None, f_res=None, 
title=None, save_path=default_save_path, collect=None):
    
    """
    finds the drive frequency of the qubit
    
    :param q_name: name of the qubit in dict_cfg
    :param probe_length: length of the probe pulse [us]
    :param gain: gain of the probe pulse [a.u.]
    :param reps: reps to be averaged over (int)
    :param f_start: start frequency [MHz]
    :param f_stop: stop frequency [MHz]
    :param f_expts: # of frequency points (int)
    :param dc: bias voltage [V]
    :param datatracker: string of data location. generated in DCROFrequency1D
    ...
    
    :params plot, save, trial: see plotsave1d function above
    :param collect: if True, returns the data
    
    """
    f_start = f_tuple[0] ; f_stop = f_tuple[1] ; f_expts = f_tuple[2]

    if use_dbm == True:
        gain=int(dbm2gain(gain, (f_start+f_stop)/2, 2, 3))

    expt_cfg={"reps": reps,"rounds":1,
              "probe_length":soccfg.us2cycles(probe_length, gen_ch=2), "qubit_gain":gain, "voltage":dc
             }

    dict_cfg = loadfrompickle(q_name, path=pkl_path)
    config={**dict_cfg, **expt_cfg}

    if config["threshold"] is not None:
        corr=config["readout_length"]
    else:
        corr=1

    if f_res != None:
        config["f_res"]=f_res
    else:
        f_res = find_nearest(config["dc_arr"], dc)
        config["f_res"]=f_res
            
    amps=[]
    f_range = np.linspace(f_start,f_stop,f_expts)
    for i in tqdm(f_range):
            config["qubit_freq"]=i
            qspec=PulseProbeSpectroscopyProgram1D(soccfg, config)
            avgi, avgq = qspec.acquire(soc, threshold=None, angle=None, load_pulses=True, progress=False, debug=False)
            amps.append((avgi[0][0]+1j*avgq[0][0])*corr)
    soc.reset_gens()

    yampl = np.abs(amps)
    yangl = np.angle(amps)

    if title==None:
        title="Qubit Spectroscopy 1D"

    f_dict = {'frange':(f_range, "Qubit Frequency", "MHz")}
    amplitudes_tuple = (yampl, "Amplitude", "DAC Units", "Amplitude") ; angles_tuple = (yangl, "Angle", "Radians", "Angle") 
    labrad1d(q_name=q_name, path=save_path, title=title, config=dict_cfg, xdict=f_dict, amplitudes=amplitudes_tuple, angles=angles_tuple)

    if collect:
        return yampl, yangl
    
class GainDCSpectroscopyProgram(NDAveragerProgram):
    #2D Scan over Qubit Frequency and Gain
    def initialize(self):
        cfg=self.cfg

        #set up generator channels (allocating tProc memory). Need to do for loops.
        self.declare_gen(ch=cfg["res_ch"], nqz=2, ro_ch=cfg["ro_ch"])
        self.declare_gen(ch=cfg["qubit_ch"], nqz=2, ro_ch=cfg["ro_ch"])
        self.declare_gen(ch=cfg["dc_ch"], ro_ch=cfg["ro_ch"])
        self.declare_readout(ch=cfg["ro_ch"], length=cfg["readout_length"], freq=cfg["f_res"], gen_ch=cfg["res_ch"])
        self.declare_readout(ch=cfg["ro_ch"], length=cfg["readout_length"], freq=cfg["f_res"], gen_ch=cfg["qubit_ch"])

        #static variables
        global f_res, f_ge
        f_res = self.freq2reg(cfg["f_res"], gen_ch=cfg["res_ch"], ro_ch=cfg["ro_ch"])
        f_ge = self.freq2reg(cfg["f_ge"], gen_ch=cfg["qubit_ch"], ro_ch=cfg["ro_ch"])
        phase = self.deg2reg(cfg["phase"])

        # --------------------- THE ND LOOP ---------------------
        #getting registers for loop
        self.qubit_r_gain = self.get_gen_reg(cfg["qubit_ch"], "gain")
        self.dc_r_gain = self.get_gen_reg(cfg["dc_ch"], "gain")
        #loop definition
        self.add_sweep(QickSweep(self, self.qubit_r_gain, cfg["g_start"], cfg["g_stop"], cfg["g_expts"]))
        self.add_sweep(QickSweep(self, self.dc_r_gain, cfg["dc_g_start"], cfg["dc_g_stop"], cfg["dc_g_expts"]))
        
        #Setting up registers for pulses
        self.add_DRAG(ch=cfg["qubit_ch"], name="drive", length=cfg["probe_length"], sigma=cfg["probe_length"]/4,
                      delta=cfg["delta"], alpha=cfg["alpha"])
        self.add_gauss(ch=cfg["res_ch"], name="measure", sigma=cfg["res_sigma"]/4, length=cfg["res_sigma"])

        self.set_pulse_registers(ch=cfg["qubit_ch"], style="arb", freq=f_ge, 
                                 gain=cfg["g_start"], waveform="drive", phase=0)
        self.set_pulse_registers(ch=cfg["res_ch"], style="flat_top", waveform="measure", freq=f_res, length=cfg["res_length"], 
                                 gain=cfg["res_gain"], phase=phase)

        #DC waveform (running in background for whole pulse sequence)
        waveform = np.ones((cfg["probe_length"]+cfg["res_length"]+cfg["drive_ro_delay"])*16)*cfg["voltage"]
        self.add_pulse(ch=cfg["dc_ch"], name="dc", idata=waveform)
        self.set_pulse_registers(ch=cfg["dc_ch"], outsel='input', style='arb', phase=0, 
                                 freq=0, waveform='dc', gain=cfg["dc_g_start"], mode='oneshot')
        
        self.synci(200)

    def body(self):
        cfg=self.cfg
        
        self.pulse(ch=cfg["dc_ch"])
           
        #drive pulse
        self.pulse(ch=self.cfg["qubit_ch"])
        self.wait_all(cfg["drive_ro_delay"])
        #measure pulse
        self.trigger(adcs=[cfg["ro_ch"]],
                     pins=[0],
                     adc_trig_offset=cfg["adc_trig_offset"], t=int(cfg["probe_length"]+cfg["drive_ro_delay"]))
        
        self.pulse(ch=cfg["res_ch"], t=int(cfg["probe_length"]+cfg["drive_ro_delay"]))
        self.wait_all(cfg["stl_time"])
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))

def QubitSpectroscopyGainDC(q_name=None, pkl_path=default_param_path,
probe_length=None, reps=300, f_res=None, f_ge=None, dc_tuple=None, g_tuple=None, use_dbm=False, 
title=None, path="Jeffrey_Q_B.dir/", collect=None):
    
    """
    finds the drive frequency of the qubit
    
    :param q_name: name of the qubit in dict_cfg
    :param probe_length: length of the probe pulse [us]
    :param gain: gain of the probe pulse [a.u.]
    :param reps: reps to be averaged over (int)
    :param f_start: start frequency [MHz]
    :param f_stop: stop frequency [MHz]
    :param f_expts: # of frequency points (int)
    :param dc: bias voltage [V]
    :param datatracker: string of data location. generated in DCROFrequency1D
    ...
    
    :params plot, save, trial: see plotsave1d function above
    :param collect: if True, returns the data
    
    """
    
    dc_start=dc2gain(6, dc_tuple[0]) ; dc_stop=dc2gain(6, dc_tuple[1]) ; dc_g_expts=dc_tuple[2]
    dc_g_start = int((dc_start/dc_stop)*32766) ; dc_g_stop=32766 #Can't iterate over waveform in ASM time, need to go over gain instead.
    g_start=g_tuple[0] ; g_stop=g_tuple[1] ; g_expts=g_tuple[2]
    units="DAC Units"

    if use_dbm:
        g_start=dbm2gain(g_start, (f_ge)/2, 2, 3)
        g_stop=dbm2gain(g_stop, (f_ge)/2, 2, 3)
        units="dBm"

    expt_cfg={"reps": reps,
            "rounds":1,
            "f_ge":f_ge,
            "g_start":g_start,
            "g_stop":g_stop,
            "g_expts":g_expts,
            "dc_g_start": dc_g_start,
            "dc_g_stop": dc_g_stop,
            "dc_g_expts": dc_g_expts,
            "voltage":dc_stop,
            
              "probe_length":soccfg.us2cycles(probe_length, gen_ch=2)
             }

    dict_cfg = loadfrompickle(q_name, path=pkl_path)
    config={**dict_cfg, **expt_cfg}

    if config["threshold"] is not None:
        corr=config["readout_length"]
    else:
        corr=1

    if f_res != None:
        config["f_res"]=f_res

    if f_ge != None:
        config["f_ge"]=f_ge

    qspec=GainDCSpectroscopyProgram(soccfg, config)
    expt_pts, avgi, avgq = qspec.acquire(soc, threshold=None, angle=None, load_pulses=True, progress=False, debug=False)
    results = (avgi[0][0]+1j*avgq[0][0])*corr
    zampl = np.abs(results)
    zangl = np.angle(results)

    dc_range = np.linspace(dc_tuple[0], dc_tuple[1], dc_g_expts)
    g_range = np.linspace(g_start, g_stop, g_expts)

    if title==None:
        title = "QB Spectroscopy Gain DC"
    
    dc_dict = {'dcrange':(dc_range, "DC Voltage", "V")}
    g_dict = {'grange':(g_range, "Gain", units)}
    amplitudes_tuple = (zampl, "Amplitude", "DAC Units", "Amplitude")
    angles_tuple = (zangl, "Angle", "Radians", "Angle")
    labrad2d(q_name=q_name, path=path, title=title, config=dict_cfg, xdict=dc_dict, ydict=g_dict, amplitudes=amplitudes_tuple, angles=angles_tuple)
    
    if collect:
        return zampl, zangl

    soc.reset_gens()

class FreqGainSpectroscopyProgram(NDAveragerProgram):
    #2D Scan over Qubit Frequency and Gain
    def initialize(self):
        cfg=self.cfg

        #set up generator channels (allocating tProc memory)
        self.declare_gen(ch=cfg["res_ch"], nqz=2, ro_ch=cfg["ro_ch"])
        self.declare_gen(ch=cfg["qubit_ch"], nqz=2, ro_ch=cfg["ro_ch"])
        self.declare_readout(ch=cfg["ro_ch"], length=cfg["readout_length"], freq=cfg["f_res"], gen_ch=cfg["res_ch"])
        self.declare_readout(ch=cfg["ro_ch"], length=cfg["readout_length"], freq=cfg["f_res"], gen_ch=cfg["qubit_ch"])

        #static variables
        global f_res, phase
        f_res = self.freq2reg(cfg["f_res"], gen_ch=cfg["res_ch"], ro_ch=cfg["ro_ch"])
        phase = self.deg2reg(cfg["phase"])

        # --------------------- THE ND LOOP ---------------------
        #variables for loop
        f_ge_start = self.freq2reg(cfg["f_ge_start"], gen_ch=cfg["qubit_ch"], ro_ch=cfg["ro_ch"])
        f_ge_stop = self.freq2reg(cfg["f_ge_stop"], gen_ch=cfg["qubit_ch"], ro_ch=cfg["ro_ch"])
        #getting registers for loop
        self.qubit_r_freq = self.get_gen_reg(cfg["qubit_ch"], "freq")
        self.qubit_r_gain = self.get_gen_reg(cfg["qubit_ch"], "gain")
        #loop definition
        self.add_sweep(QickSweep(self, self.qubit_r_freq, f_ge_start, f_ge_stop, cfg["f_ge_expts"]))
        self.add_sweep(QickSweep(self, self.qubit_r_gain, cfg["g_start"], cfg["g_stop"], cfg["g_expts"]))
        
        #Setting up registers for pulses
        self.add_DRAG(ch=cfg["qubit_ch"], name="drive", length=cfg["probe_length"], sigma=cfg["probe_length"]/4,
                      delta=cfg["delta"], alpha=cfg["alpha"])
        self.add_gauss(ch=cfg["res_ch"], name="measure", sigma=cfg["res_sigma"]/4, length=cfg["res_sigma"])

        self.set_pulse_registers(ch=cfg["qubit_ch"], style="arb", freq=f_ge_start, 
                                 gain=cfg["g_start"], waveform="drive", phase=0)
        self.set_pulse_registers(ch=cfg["res_ch"], style="flat_top", waveform="measure", freq=f_res, length=cfg["res_length"], 
                                 gain=cfg["res_gain"], phase=phase)

        #DC waveform (running in background for whole pulse sequence)
        waveform = np.ones((cfg["probe_length"]+cfg["res_length"]+cfg["drive_ro_delay"])*16)*cfg["voltage"]
        self.add_pulse(ch=cfg["dc_ch"], name="dc", idata=waveform)
        self.set_pulse_registers(ch=cfg["dc_ch"], outsel='input', style='arb', phase=0, 
                                 freq=0, waveform='dc', gain=32767, mode='oneshot')
        
        self.synci(200)

    def body(self):
        cfg=self.cfg
        
        self.pulse(ch=cfg["dc_ch"])
           
        #drive pulse
        self.pulse(ch=self.cfg["qubit_ch"])
        self.wait_all(cfg["drive_ro_delay"])
        #measure pulse
        self.trigger(adcs=[cfg["ro_ch"]],
                     pins=[0],
                     adc_trig_offset=cfg["adc_trig_offset"], t=int(cfg["probe_length"]+cfg["drive_ro_delay"]))
        
        self.pulse(ch=cfg["res_ch"], t=int(cfg["probe_length"]+cfg["drive_ro_delay"]))
        self.wait_all(cfg["stl_time"])
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))

def QubitSpectroscopyFreqGain(q_name=None, pkl_path=default_param_path,
probe_length=None, reps=300, dc=None, f_res=None, f_tuple=None, g_tuple=None, use_dbm=False,
title=None, path="Jeffrey_Q_B.dir/", collect=None):
    
    """
    finds the drive frequency of the qubit
    
    :param q_name: name of the qubit in dict_cfg
    :param probe_length: length of the probe pulse [us]
    :param gain: gain of the probe pulse [a.u.]
    :param reps: reps to be averaged over (int)
    :param f_start: start frequency [MHz]
    :param f_stop: stop frequency [MHz]
    :param f_expts: # of frequency points (int)
    :param dc: bias voltage [V]
    :param datatracker: string of data location. generated in DCROFrequency1D
    ...
    
    :params plot, save, trial: see plotsave1d function above
    :param collect: if True, returns the data
    
    """
    
    f_ge_start=f_tuple[0] ; f_ge_stop=f_tuple[1] ; f_ge_expts=f_tuple[2]
    g_start=g_tuple[0] ; g_stop=g_tuple[1] ; g_expts=g_tuple[2]
    units="DAC Units"

    if use_dbm:
        g_start=dbm2gain(g_start, (f_ge_start+f_ge_stop)/2, 2, 3)
        g_stop=dbm2gain(g_stop, (f_ge_start+f_ge_stop)/2, 2, 3)
        units="dBm"

    expt_cfg={"reps": reps,"rounds":1,
            "f_ge_start":f_ge_start,
            "f_ge_stop":f_ge_stop,
            "f_ge_expts":f_ge_expts,
            "g_start":g_start,
            "g_stop":g_stop,
            "g_expts":g_expts,
            "voltage":dc2gain(6, dc),

              "probe_length":soccfg.us2cycles(probe_length, gen_ch=2)
             }

    dict_cfg = loadfrompickle(q_name, path=pkl_path)
    config={**dict_cfg, **expt_cfg}
    if config["threshold"] is not None:
        corr=config["readout_length"]
    else:
        corr=1
    if f_res != None:
        config["f_res"]=f_res


    qspec=FreqGainSpectroscopyProgram(soccfg, config)
    expt_pts, avgi, avgq = qspec.acquire(soc, threshold=None, angle=None, load_pulses=True, progress=False, debug=False)
    results = (avgi[0][0]+1j*avgq[0][0])*corr
    zampl = np.abs(results)
    zangl = np.angle(results)

    f_range = np.linspace(f_ge_start, f_ge_stop, f_ge_expts)
    g_range = np.linspace(g_start, g_stop, g_expts)

    if title==None:
        title = "QB Spectroscopy"

    f_dict = {'frange':(f_range, "Qubit Frequency", "MHz")}
    g_dict = {'grange':(g_range, "Gain", units)}
    amplitudes_tuple = (zampl, "Amplitude", "DAC Units", "Amplitude")
    angles_tuple = (zangl, "Angle", "Radians", "Angle")
    labrad2d(q_name=q_name, path=path, title=title, config=dict_cfg, xdict=f_dict, ydict=g_dict, amplitudes=amplitudes_tuple, angles=angles_tuple)

    if collect:
        return zampl, zangl
    
    soc.reset_gens()

class NewPulseProbeZPASpectroscopy(RAveragerProgram):
    #Sweep DC flattop voltage externally, Frequency internally
    def initialize(self):
        cfg = self.cfg
        self.declare_gen(ch=cfg["qubit_ch"], nqz=2) #should be nqz=2 in real runs
        self.declare_gen(ch=cfg["res_ch"], nqz=2) #should be nqz=2 in real runs
        self.declare_gen(ch=cfg["dc_ch"], nqz=1)
        for ch in [0]:
            self.declare_readout(ch=ch, length=cfg["readout_length"],
                                 freq=cfg["f_res"], gen_ch=cfg["res_ch"])
        global f_res, f_ge
        f_res = self.freq2reg(cfg["f_res"], gen_ch=cfg["res_ch"], ro_ch=0)
        f_ge = self.freq2reg(cfg["f_ge"], gen_ch=cfg["qubit_ch"], ro_ch=0)

        global phase
        phase=self.deg2reg(cfg["phase"])

        #setting up drive + readout
        self.add_DRAG(ch=cfg["qubit_ch"], name="drive", length=cfg["probe_length"], sigma=cfg["probe_length"]/4,
                      delta=cfg["delta"], alpha=cfg["alpha"])
        self.set_pulse_registers(ch=cfg["qubit_ch"], style="arb", freq=cfg["start"], 
                                 gain=cfg["qubit_gain"], waveform="drive", phase=0)
        
        self.add_gauss(ch=cfg["res_ch"], name="measure", sigma=cfg["res_sigma"]/4, length=cfg["res_sigma"])
        self.set_pulse_registers(ch=cfg["res_ch"], style="flat_top", waveform="measure", freq=f_res, length=cfg["res_length"], 
                                 gain=cfg["res_gain"], phase=phase)
        
        #setting up DC - must be done in this cell to be updated
        drivelen = (cfg["ramp_delay"] + cfg["probe_length"])*16 #add ramp delay so we can ramp up correctly
        rolen = (cfg["drive_ro_delay"] + cfg["res_length"] + 50)*16 #50 clock cycles to account for trigger setting
        
        #DC waveform (running in background for whole pulse sequence)
        waveform = np.hstack((np.ones(drivelen)*cfg["topamp"], np.ones(rolen)*cfg["baseamp"]))
        self.add_pulse(ch=cfg["dc_ch"], name="dc", idata=waveform)
        self.set_pulse_registers(ch=cfg["dc_ch"], outsel='input', style='arb', phase=0, 
                                 freq=0, waveform='dc', gain=32767, mode='oneshot')
        
        #parameters for RAverager sweep
        self.q_pg=self.ch_page(self.cfg["qubit_ch"])
        self.q_freq=self.sreg(cfg["qubit_ch"], "freq")
        
        self.sync_all(200)
        
    def body(self):
        cfg = self.cfg
        
        soc.set_all_clks()
        
        #calculate delay from readout (qubit pulse + ramp up + ro delay)
        readout_start =  cfg["ramp_delay"] + cfg["probe_length"] + cfg["drive_ro_delay"] #add 32 clock cycle (74 ns) ramp-up time so we can ramp up correctly
        
        self.pulse(ch=cfg["dc_ch"])
        
        self.waiti(cfg["qubit_ch"],cfg["ramp_delay"])
        self.pulse(ch=cfg["qubit_ch"])
        
        self.trigger(adcs=[0],
            pins=[0],
            adc_trig_offset=cfg["adc_trig_offset"])
                
        self.waiti(cfg["res_ch"],readout_start)
        self.pulse(ch=cfg["res_ch"])

        self.wait_all(cfg["stl_time"])
        self.sync_all(cfg["stl_time"])
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))
    
    def update(self):
        self.mathi(self.q_pg, self.q_freq, self.q_freq, '+', self.cfg["step"])

def NewZPASpectroscopy(q_name=None, pkl_path=default_param_path,
                    dc_bias=0, probe_length=None, res_gain=None, qubit_gain=None, f_res=None, 
                    f_tuple=None, zpa_tuple=None, reps=300,
                path="Jeffrey_Q_B.dir/", title=None, collect=None):

    if q_name==None or probe_length==None or f_tuple==None or zpa_tuple==None:
        raise Exception("You must specify q_name, probe_length, gain, f_tuple, zpa_tuple")
    
    
    f_start = f_tuple[0] ; f_stop = f_tuple[1] ; f_expts = f_tuple[2]
    f_range = np.linspace(f_start, f_stop, f_expts)
    
    zpa_start = zpa_tuple[0] ; zpa_stop = zpa_tuple[1] ; zpa_expts = zpa_tuple[2]
    yvar = np.linspace(zpa_start, zpa_stop, zpa_expts)
    yvarname = "Flat Top Voltage"
    units = "V"

    dict_cfg = loadfrompickle(q_name, pkl_path)
    expt_cfg={"reps":reps,
              "rounds":1,
              "baseamp" : dc2gain(dict_cfg["dc_ch"], dc_bias),

              #Experiment variables
              "voltage":dc_bias,
              "probe_length":soccfg.us2cycles(probe_length),
              
              #RAverager Specific variable names
              "expts":f_expts, #hidden variable, used in the assembly loop
              "start": soccfg.freq2reg(f_start), #hidden variable, used in the assembly loop
              "step": soccfg.freq2reg(f_stop), #needs to be called step
             }
    config = {**dict_cfg, **expt_cfg}
    if config["threshold"] is not None:
        corr=config["readout_length"]
    else:
        corr=1
    if f_res != None:
        config["f_res"]=f_res
    else:
        f_res = find_nearest(config["dc_arr"], dc)
        config["f_res"]=f_res-0.7
        
    if res_gain != None:
        config["res_gain"]=res_gain
    if qubit_gain != None:
        config["qubit_gain"]=qubit_gain
    
    config["res_ch"]=2
    config["qubit_ch"]=0
    
    #Actual experiment run
    results=[]
    soc.reset_gens()
    for i in yvar:
        config["topamp"]= dc2gain(dict_cfg["dc_ch"], i)
        prog = NewPulseProbeZPASpectroscopy(soccfg, config)
        expt_pts, avgi, avgq = prog.acquire(soc, load_pulses=True, progress=False)
        results.append((avgi[0][0]+1j*avgq[0][0])*corr)
    soc.reset_gens()
    zampl = np.abs(results)
    zangl = np.angle(results)
    

    if title==None:
        title = "ZPA Spectroscopy"
    
    f_dict = {'frange':(f_range, "Qubit Frequency", "MHz")}
    g_dict = {'zparange':(yvar, yvarname, units)}
    amplitudes_tuple = (zampl, "Amplitude", "DAC Units", "Amplitude")
    angles_tuple = (zangl, "Angle", "Radians", "Angle")
    labrad2d(q_name=q_name, path=path, title=title, trial=trial, config=dict_cfg, xdict=f_dict, ydict=g_dict, amplitudes=amplitudes_tuple, angles=angles_tuple)

    if collect:
        return zampl, zangl

class RabiProgram(AveragerProgram):
    def initialize(self):
        cfg=self.cfg

        self.declare_gen(ch=cfg["qubit_ch"], nqz=2) #Qubit
        self.declare_gen(ch=cfg["res_ch"], nqz=2)

        for ch in [0]:
            self.declare_readout(ch=ch, length=cfg["readout_length"],
                                 freq=cfg["f_res"], gen_ch=cfg["res_ch"])

        #Setting up qubit pulse
        f_ge=self.freq2reg(cfg["f_ge"], gen_ch=cfg["qubit_ch"])
        self.add_DRAG(ch=cfg["qubit_ch"], name="drive", length=cfg["qubit_length"], sigma=cfg["qubit_length"]/4,
                      delta=cfg["delta"], alpha=cfg["alpha"])
        self.set_pulse_registers(ch=cfg["qubit_ch"], style="arb", waveform="drive", freq=f_ge, gain=cfg["qubit_gain"], phase=0)

        #Setting up resonator pulse
        phase=self.deg2reg(cfg["phase"])
        f_res=self.freq2reg(cfg["f_res"], gen_ch=cfg["res_ch"])
        self.add_gauss(ch=cfg["res_ch"], name="measure", sigma=cfg["res_sigma"]/4, length=cfg["res_sigma"])
        self.set_pulse_registers(ch=cfg["res_ch"], style="flat_top", waveform="measure", freq=f_res, length=cfg["res_length"], 
                                 gain=cfg["res_gain"], phase=phase)
        
        #Setting up dc pulse
        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=4, mode="periodic")
        
        self.synci(200)
        
    def body(self):
        cfg=self.cfg
        
        self.pulse(ch=cfg["dc_ch"])
        
        self.pulse(ch=cfg["qubit_ch"])
        
        self.trigger(adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=cfg["adc_trig_offset"])
        
        self.pulse(ch=cfg["res_ch"], t=cfg["qubit_length"]+cfg["drive_ro_delay"])
        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))

class NDRabiProgramGL(NDAveragerProgram):
    def initialize(self):
        cfg=self.cfg

        self.declare_gen(ch=cfg["qubit_ch"], nqz=2) #Qubit
        self.declare_gen(ch=cfg["res_ch"], nqz=2)

        for ch in [0,1]:
            self.declare_readout(ch=ch, length=cfg["readout_length"],
                                 freq=cfg["f_res"], gen_ch=cfg["res_ch"])
        
        #setting up resonator pulse
        phase=self.deg2reg(cfg["phase"])
        f_res=self.freq2reg(cfg["f_res"], gen_ch=cfg["qubit_ch"], ro_ch=0)
        self.add_gauss(ch=cfg["res_ch"], name="measure", sigma=cfg["res_sigma"]/4, length=cfg["res_sigma"])
        self.set_pulse_registers(ch=cfg["res_ch"], style="flat_top", waveform="measure", freq=f_res, length=cfg["res_length"], 
                                 gain=cfg["res_gain"], phase=phase)

        #setting up qubit pulse
        f_ge=self.freq2reg(cfg["f_ge"], gen_ch=cfg["qubit_ch"])
        self.add_DRAG(ch=cfg["qubit_ch"], name="drive", length=cfg["qubit_length"], sigma=cfg["qubit_length"]/4,
                      delta=cfg["delta"], alpha=cfg["alpha"])
        self.set_pulse_registers(ch=cfg["qubit_ch"], style="arb", waveform="drive", freq=f_ge, gain=cfg["g_start"], phase=0)
        
        #setting up scan
        self.res_r_gain = self.get_gen_reg(cfg["qubit_ch"], "gain")
        self.res_r_gain_update = self.new_gen_reg(cfg["qubit_ch"], init_val=cfg["g_start"], name="gain_update") 
        self.add_sweep(QickSweep(self, self.res_r_gain_update, cfg["g_start"], cfg["g_stop"], cfg["g_expts"]))
        
        #Seting up dc pulse
        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=4, mode="periodic")
        
        self.synci(200)
        
    def body(self):
        cfg=self.cfg
        
        self.pulse(ch=cfg["dc_ch"])

        self.res_r_gain.set_to(self.res_r_gain_update)
        self.pulse(ch=cfg["qubit_ch"])
        
        self.trigger(adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=cfg["adc_trig_offset"])
        
        self.pulse(ch=cfg["res_ch"], t=cfg["qubit_length"]+cfg["drive_ro_delay"])
        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))

class NDRabiProgramFL(NDAveragerProgram):
    def initialize(self):
        cfg=self.cfg

        self.declare_gen(ch=cfg["qubit_ch"], nqz=2) #Qubit
        self.declare_gen(ch=cfg["res_ch"], nqz=2)

        for ch in [0]:
            self.declare_readout(ch=ch, length=cfg["readout_length"],
                                 freq=cfg["f_res"], gen_ch=cfg["res_ch"])
        
        #resonator setup
        phase=self.deg2reg(cfg["phase"])
        f_res=self.freq2reg(cfg["f_res"], gen_ch=cfg["qubit_ch"], ro_ch=0)
        self.add_gauss(ch=cfg["res_ch"], name="measure", sigma=cfg["res_sigma"]/4, length=cfg["res_sigma"])
        self.set_pulse_registers(ch=cfg["res_ch"], style="flat_top", waveform="measure", freq=f_res, length=cfg["res_length"], 
                                 gain=cfg["res_gain"], phase=phase)

        
        self.add_DRAG(ch=cfg["qubit_ch"], name="drive", length=cfg["qubit_length"], sigma=cfg["qubit_length"]/4,
                      delta=cfg["delta"], alpha=cfg["alpha"])
        self.set_pulse_registers(ch=cfg["qubit_ch"], style="arb", waveform="drive", freq=cfg["f_start"], gain=cfg["qubit_gain"], phase=0)
        
        self.res_r_freq = self.get_gen_reg(cfg["qubit_ch"], "freq")
        self.res_r_freq_update = self.new_gen_reg(cfg["qubit_ch"], init_val=cfg["f_start"], name="freq_update") 
        self.add_sweep(QickSweep(self, self.res_r_freq_update, cfg["f_start"], cfg["f_stop"], cfg["f_expts"]))
        
        #DC setup
        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=4, mode="periodic")

        self.synci(200)
        
    def body(self):
        cfg=self.cfg
        
        self.pulse(ch=cfg["dc_ch"])
        
        self.res_r_freq.set_to(self.res_r_freq_update)
        self.pulse(ch=cfg["qubit_ch"])
        
        self.trigger(adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=cfg["adc_trig_offset"])
        
        self.pulse(ch=cfg["res_ch"], t=cfg["qubit_length"]+cfg["drive_ro_delay"])
        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))
    
def Rabi(q_name=None, pkl_path=default_param_path,
        qubit_gain=None, qubit_length=None, l_start=None, l_stop=None, l_expts=None,
        g_start=None, g_stop=None, g_expts=None, f_start=None, f_stop=None, f_expts=None,
        reps=None, dc=None, plot=None, save=None, trial=None, collect=None, fit=None):
    
    """
    finds the length of the pi pulse with a given amplitude
    
    :param q_name: name of the qubit in dict_cfg
    :param gain: gain of the drive pulse [a.u.]
    :param reps: reps to be averaged over (int)
    :param start: start length [us]
    :param stop: stop length [us]
    :param expts: # of length points (int)
    
    All lengths are increased by ~10ns, which is the minimum length
    ...
    
    :params plot, save, trial: see plotsave1d function above
    :param collect: if True, returns the data. If false, returns pi pulse length.
    
    """

    if q_name==None or reps==None or dc==None:
        raise Exception("You must specify q_name, reps, and dc")

    if qubit_gain!=None and qubit_length!=None:
        raise Exception("You can specify gain (Length Rabi) or length (Amplitude Rabi), not both.")
    
    if qubit_gain==None and qubit_length==None:
        fit=None
    
    dict_cfg = loadfrompickle(q_name, path=pkl_path)
    
    if qubit_gain!=None:
        expt_cfg={
            "qubit_gain":qubit_gain,
            "l_start":soccfg.us2cycles(l_start), "l_stop":soccfg.us2cycles(l_stop), "l_expts":l_expts, "reps":reps, "voltage":dc
           }
        config={**dict_cfg, **expt_cfg}
        xvar=np.linspace(l_start, l_stop, l_expts)*1000+9.3
        xvarname="Pulse length (ns)"

    elif qubit_length!=None and g_start!=None and l_start==None:
        expt_cfg={
            "qubit_length":soccfg.us2cycles(qubit_length),
            "g_start":g_start, "g_stop":g_stop, "g_expts":g_expts, "reps":reps, "voltage":dc
           }
        config={**dict_cfg, **expt_cfg}
        xvar=np.linspace(g_start, g_stop, g_expts)
        xvarname="Gain"
    
    elif g_start!=None and l_start!=None:
        expt_cfg={
            "l_start":soccfg.us2cycles(l_start), "l_stop":soccfg.us2cycles(l_stop), "l_expts":l_expts,
            "g_start":g_start, "g_stop":g_stop, "g_expts":g_expts, "reps":reps, "voltage":dc
           }
        config={**dict_cfg, **expt_cfg}
        xvar=np.linspace(g_start, g_stop, g_expts)
        xvarname="Gain"
        yvar=np.linspace(l_start, l_stop, l_expts)
        yvarname="Pulse length (us)"

    elif l_start!=None and f_start!=None:
        expt_cfg={
            "l_start":soccfg.us2cycles(l_start), "l_stop":soccfg.us2cycles(l_stop), "l_expts":l_expts,
            "f_start":soccfg.freq2reg(f_start), "f_stop":soccfg.freq2reg(f_stop), "f_expts":f_expts, "reps":reps, "voltage":dc
        }
        config={**dict_cfg, **expt_cfg}
        xvar=np.linspace(f_start, f_stop, f_expts)
        xvarname="Frequency (MHz)"
        yvar=np.linspace(l_start, l_stop, l_expts)
        yvarname="Pulse length (us)"
    
    if config["threshold"] is not None:
        corr=config["readout_length"]
    else:
        corr=1
    results=[]

    if qubit_gain!=None:
        for l in tqdm(np.linspace(config["l_start"], config["l_stop"], config["l_expts"])+4):
            config["qubit_length"]=l
            rabi=RabiProgram(soccfg, config)
            avgi,avgq = rabi.acquire(soc, threshold=config["threshold"], load_pulses=True, progress=False,debug=False)
            results.append(np.sqrt(avgi[0][0]**2+avgq[0][0]**2)*corr)
    
    elif qubit_length!=None and g_start!=None and l_start==None:
        for l in tqdm(np.linspace(config["g_start"], config["g_stop"], config["g_expts"])):
            config["qubit_gain"]=int(l)
            rabi=RabiProgram(soccfg, config)
            avgi,avgq = rabi.acquire(soc, threshold=config["threshold"], load_pulses=True, progress=False,debug=False)
            results.append(np.sqrt(avgi[0][0]**2+avgq[0][0]**2)*corr)
    elif g_start!=None and l_start!=None:
        for l in tqdm(np.linspace(config["l_start"], config["l_stop"], config["l_expts"])+4):
            config["qubit_length"]=l
            rabi=NDRabiProgramGL(soccfg, config)
            expt_pts, avgi, avgq = rabi.acquire(soc, threshold=config["threshold"], load_pulses=True, progress=False,debug=False)
            results.append(np.sqrt(avgi[0][0]**2+avgq[0][0]**2)*corr)
    elif l_start!=None and f_start!=None:
        for l in tqdm(np.linspace(config["l_start"], config["l_stop"], config["l_expts"])+4):
            config["qubit_length"]=l
            rabi=NDRabiProgramFL(soccfg, config)
            expt_pts, avgi, avgq = rabi.acquire(soc, threshold=config["threshold"], load_pulses=True, progress=False,debug=False)
            results.append(np.sqrt(avgi[0][0]**2+avgq[0][0]**2)*corr)

    
    if fit:
        try:
            minvalue=np.min(results)
            maxvalue=np.max(results)
            amplitude=maxvalue-minvalue
            maxsin=np.where(results>maxvalue-0.2*amplitude)[0][0]
            try:
                minsin=np.where(results<minvalue+0.2*amplitude)[0][np.where(results<minvalue+0.2*amplitude)[0]>maxsin][0]
            except:
                minsin=np.where(results<minvalue+0.2*amplitude)[0][0]

            p_guess = [amplitude/2,np.pi/(minsin-maxsin),np.pi/2,np.mean(results)]
            p_opt, p_cov = curve_fit(sinfunc, np.arange(config["expts"]), results, p0 = p_guess)

            fitfunc=sinfunc(np.arange(config["expts"]),p_opt[0],p_opt[1],p_opt[2],p_opt[3])
            try:
                step=(np.linspace(l_start, l_stop, expts)[20]-np.linspace(l_start, l_stop, expts)[0])/20
            except:
                step=(np.linspace(g_start, g_stop, expts)[20]-np.linspace(g_start, g_stop, expts)[0])/20
            pipulse = np.pi/p_opt[1]*step
            
        except:
            fitfunc=None
            pipulse=0
            print("Fit did not converge")
    else:
        fitfunc=None
    
    if qubit_length!=None or qubit_gain!=None:
        plotsave1d(plot=plot, save=save, q_name=q_name, title="Rabi", trial=trial, xvar=xvar,
                  xvarname=xvarname, ydata=results,ydataname="Qubit population", fitfunc=fitfunc, config=config)
    
    else:
        plotsave2d(plot=plot, save=save, q_name=q_name, title="Rabi", trial=trial, xvar=xvar,
                  xvarname=xvarname, yvar=yvar, yvarname=yvarname, zdata=results, zdataname="Qubit Population",
                   fitfunc=fitfunc, config=config)

    soc.reset_gens()
    
    if collect:
        return results
    else:
        if fit:
            return np.abs(pipulse)

class Pi2AmplitudeRabiProgram(AveragerProgram):
    def initialize(self):
        cfg=self.cfg

        self.declare_gen(ch=cfg["qubit_ch"], nqz=2) #Qubit
        self.declare_gen(ch=cfg["res_ch"], nqz=2)
        for ch in [0,1]:
            self.declare_readout(ch=ch, length=cfg["readout_length"],
                                 freq=cfg["f_res"], gen_ch=cfg["res_ch"])
        
        global phase
        phase=self.deg2reg(cfg["phase"])
        self.add_DRAG(ch=cfg["qubit_ch"], name="drive", length=cfg["qubit_length"], sigma=cfg["qubit_length"]/4,
                      delta=cfg["delta"], alpha=cfg["alpha"])
        self.add_gauss(ch=cfg["qubit_ch"], name="measure", sigma=cfg["res_sigma"]/4, length=cfg["res_sigma"])
        
        self.synci(200)
        
    def body(self):
        cfg=self.cfg
        
        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=4, mode="periodic")
        self.pulse(ch=cfg["dc_ch"])

        f_res=self.freq2reg(cfg["f_res"], gen_ch=cfg["qubit_ch"], ro_ch=0)
        f_ge=self.freq2reg(cfg["f_ge"], gen_ch=cfg["qubit_ch"])
        
        self.set_pulse_registers(ch=cfg["qubit_ch"], style="arb", waveform="drive", freq=f_ge, gain=cfg["qubit_gain"], phase=0)
        self.pulse(ch=cfg["qubit_ch"])
        self.pulse(ch=cfg["qubit_ch"])

        self.sync_all(cfg["drive_ro_delay"])

        self.pulse(ch=cfg["dc_ch"])
        
        self.set_pulse_registers(ch=cfg["res_ch"], style="flat_top", waveform="measure", freq=f_res, length=cfg["res_length"], 
                                 gain=cfg["res_gain"], phase=phase)
        
        self.trigger(adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=cfg["adc_trig_offset"])
        
        self.pulse(ch=cfg["res_ch"])
        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))

def Pi2AmplitudeRabi(q_name=None, length=None, g_start=None, g_stop=None, g_expts=None, reps=None, dc=None, plot=None, save=None, trial=None, collect=None, fit=None):
    
    """
    finds the amplitude of the pi/2 pulse with a given length
    
    :param q_name: name of the qubit in dict_cfg
    :param length: length of the drive pulse [us]
    :param reps: reps to be averaged over (int)
    :param g_start: start gain [a.u.]
    :param g_stop: stop gain [a.u.]
    :param g_expts: # of gain points (int)
    :param dc: dc voltage [V]
    ...
    
    :params plot, save, trial: see plotsave1d function above
    :param collect: if True, returns the data. If false, returns pi/2 pulse amplitude.
    
    """
    
    if q_name==None or length==None or g_start==None or g_stop==None or g_expts==None or reps==None or dc==None:
        raise Exception("You must specify q_name, length, g_start, g_stops, g_expts, reps, and dc.")

    expt_cfg={
            "qubit_length":soccfg.us2cycles(length), "reps":reps, "voltage":dc
           }
    dict_cfg = loadfrompickle(q_name)
    config={**dict_cfg, **expt_cfg}

    if config["threshold"] is not None:
        corr=config["readout_length"]
    else:
        corr=1
    results=[]
    for l in tqdm(np.linspace(g_start, g_stop, g_expts)):
        config["qubit_gain"]=int(l)
        rabi=Pi2AmplitudeRabiProgram(soccfg, config)
        avgi,avgq = rabi.acquire(soc, threshold=config["threshold"], load_pulses=True, progress=False,debug=False)
        results.append(np.sqrt(avgi[0][0]**2+avgq[0][0]**2)*corr)
        
    #automatic fit
    if fit:
        minvalue=np.min(results)
        maxvalue=np.max(results)
        amplitude=maxvalue-minvalue
        maxsin=np.where(results>maxvalue-0.2*amplitude)[0][0]
        try:
            minsin=np.where(results<minvalue+0.2*amplitude)[0][np.where(results<minvalue+0.2*amplitude)[0]>maxsin][0]
        except:
            minsin=np.where(results<minvalue+0.2*amplitude)[0][0]

        p_guess = [amplitude/2,np.pi/(minsin-maxsin),-np.pi/2,np.mean(results)]
        try:
            p_opt, p_cov = curve_fit(sinfunc, np.arange(g_expts), results, p0 = p_guess)
            p_err = np.sqrt(np.diag(p_cov))

            fitfunc=sinfunc(np.arange(expts),p_opt[0],p_opt[1],p_opt[2],p_opt[3])
            fitvalues=p_opt
            fitvalueserr=p_err
            step=(np.linspace(g_start, g_stop, g_expts)[10]-np.linspace(g_start, g_stop, g_expts)[0])/10
            pipulse = np.pi/p_opt[1]*step
        except:
            pipulse=0
            print("Fit did not converge")
    else:
        fitfunc=None
    
    plotsave1d(plot=plot, q_name=q_name, save=save, title="Pi/2 Amplitude Rabi", trial=trial, xvar=np.linspace(g_start, g_stop, g_expts),
               xvarname="Gain", ydata=results,ydataname="Qubit population", fitfunc=fitfunc, config=config)
    
    
    if collect:
        return results
    else:
        if fit:
            return np.abs(pipulse)
    
    soc.reset_gens()

class IQProgram(NDAveragerProgram):
    """
    Function to collect the IQ data of a number of individual shots, given by reps
    """
    def initialize(self):
        cfg=self.cfg

        self.declare_gen(ch=cfg["qubit_ch"], nqz=2)
        self.declare_gen(ch=cfg["res_ch"], nqz=2)
        for ch in [0]:
            self.declare_readout(ch=ch, length=cfg["readout_length"],
                                 freq=cfg["f_res"], gen_ch=cfg["res_ch"])
        
        self.add_DRAG(ch=cfg["qubit_ch"], name="drive", length=cfg["qubit_length"], sigma=cfg["qubit_length"]/4,
                      delta=cfg["delta"], alpha=cfg["alpha"])
        self.add_gauss(ch=cfg["res_ch"], name="measure", sigma=cfg["res_sigma"]/4, length=cfg["res_sigma"])
        self.res_r_gain = self.get_gen_reg(cfg["qubit_ch"], "gain")
        self.res_r_gain_update = self.new_gen_reg(cfg["qubit_ch"], init_val=cfg["start"], name="gain_update") 
        self.add_sweep(QickSweep(self, self.res_r_gain_update, cfg["start"], cfg["pi_gain"], cfg["expts"]))
        
        self.synci(200)
        
    def body(self):
        cfg=self.cfg
        
        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=4, mode="periodic")
        
        self.pulse(ch=cfg["dc_ch"])

        f_res=self.freq2reg(cfg["f_res"], gen_ch=cfg["qubit_ch"], ro_ch=0)
        f_ge=self.freq2reg(cfg["f_ge"], gen_ch=cfg["qubit_ch"])
        
        self.set_pulse_registers(ch=cfg["qubit_ch"], style='arb', waveform="drive", phase=0, freq=f_ge, gain=0)
        self.res_r_gain.set_to(self.res_r_gain_update)
        self.pulse(ch=cfg["qubit_ch"])
        
        self.sync_all(cfg["drive_ro_delay"])

        self.set_pulse_registers(ch=cfg["res_ch"], style='flat_top', waveform='measure', freq=f_res, 
                                 phase=0, 
                                 gain=int(cfg["res_gain"]), length=cfg["res_length"])

        self.pulse(ch=cfg["dc_ch"])

        self.trigger(adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=cfg["adc_trig_offset"])
        
        self.pulse(ch=cfg["res_ch"])
        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))
        
    def acquire(self,soc, load_pulses=True, progress=False, debug=False):
        super().acquire(soc, load_pulses=load_pulses, progress=progress, debug=debug)
        return self.collect_shots()
        
    def collect_shots(self):
        shots_i0=self.di_buf[0]/self.cfg['readout_length']
        shots_q0=self.dq_buf[0]/self.cfg['readout_length']
        return shots_i0,shots_q0

def IQblobs(q_name=None, reps=None, dc=None, feats=None, plot=None, save=None, trial=None, collect=None):
    
    """
    Returns the IQ data of a number of individual shots
    :param q_name: name of the qubit in dict_cfg
    :param reps: number of data points
    :param ro_feats: list with the readout pulse parameters. If none, it just uses the default configs.
        list elements should be given in the order: [length (us), ramp length (us), frequency (MHz), gain (a.u.)]
    :param drive_feats: list with the drive pulse parameters. If none, it just uses the default configs.
        list elements should be given in the order: [DRAG parameter, gain (a.u.), length (us)]
    :param plot: if True, plots the data
    :param save: if True, saves the data
    :param collect: if True, returns the data
    """

    if q_name==None or reps==None or dc==None:
        raise Exception("You must specify q_name, reps, and dc")


    expt_cfg={
            "start":0, "expts":2, "reps":reps, "voltage":dc
           }
    dict_cfg = loadfrompickle(q_name)
    config={**dict_cfg, **expt_cfg}

    config["qubit_length"]=config["pulse_length"]
    
    if feats!=None:
        for key in feats.keys():
            config[key]=feats[key]

    config["readout_length"]=int(config["readout_length"])
    config["adc_trig_offset"]=int(config["adc_trig_offset"])
    iq = IQProgram(soccfg, config)
    avgi, avgq = iq.acquire(soc, load_pulses=True, progress=False,debug=False)
    idx=np.arange(reps)*2
    avgi0=avgi[idx]
    avgq0=avgq[idx]
    avgi1=avgi[idx+1]
    avgq1=avgq[idx+1]
    
    title = "IQ Blobs"

    if trial is None:
            trial="01"
    
    if plot is not None and plot:
        plt.scatter(avgi0, avgq0)
        plt.scatter(avgi1, avgq1)
        plt.axis("equal")
        plt.xlabel("I")
        plt.ylabel("Q")
        if plot and save:
            plt.savefig('images/{}/Stark-{}-{}-{}.pdf'.format(q_name, title, date.today(), trial))
    
    if save is not None and save:

        data0 = np.transpose(np.vstack((avgi0,avgq0)))
        data1 = np.transpose(np.vstack((avgi1,avgq1)))
        y_data = np.vstack((data0, data1))
        x_data = np.hstack(( np.zeros(reps), np.ones(reps) ))
        
        df=pd.DataFrame(y_data, index=x_data)
        df.to_csv('measurement_data/{}/Data-Stark-{}-{}-{}.csv'.format(q_name, title, date.today(), trial), mode='a', header=False)
        #np.savetxt('measurement_data/{}/Data-Stark-{}-{}-{}.csv'.format(q_name, title, date.today(), trial), iq_return, delimiter=",")
        dumptopickle('./measurement_data/{}/Settings-Stark-{}-{}-{}.pickle'.format(q_name, title, date.today(), trial), config)
        
    if collect:
        return avgi0, avgq0, avgi1, avgq1, 
    soc.reset_gens()
    
def find_threshold(q_name=None, n_datapoints=None, dc=None, feats=None, plot=None):
    """
    Subsitutes the value of Threshold and phase in config
    :param q_name: name of the qubit in dict_cfg
    :param n_datapoints: number of datapoints
    :param plot: if True, plots the data
    """
    if q_name==None or n_datapoints==None or dc==None:
        raise Exception("You must specify q_name, n_datapoints, and dc")

    threshold, theta = hist(data=IQblobs(q_name, n_datapoints, dc, feats=feats, plot=False, collect=True), plot=plot)
    writetopickle(q_name, threshold=threshold, phase=-theta*180/np.pi)
    return threshold, theta

def calc_visibility(q_name=None, n_datapoints=None, dc=None, feats=None, plot=None):
    """
    Subsitutes the value of Threshold and phase in config
    :param q_name: name of the qubit in dict_cfg
    :param n_datapoints: number of datapoints
    :param plot: if True, plots the data
    """
    if q_name==None or n_datapoints==None or dc==None:
        raise Exception("You must specify q_name, n_datapoints, and dc")


    a, b, c, d = IQblobs(q_name, n_datapoints, dc, feats=feats, plot=False, collect=True)
    initial_center=[np.mean(c),np.mean(d)]        
    return maximize_visibility(a,b,c,d, initial_center, plot=plot)

def differential_evolution(q_name=None, n_walkers=None, steps=None, thresh=None, evol=None, bounds=None, n_points=None, dc=None, plot=None, save=None, trial=None, collect=None):

    """
    Differential evolution for readout optimization, automatically updates config
    
    :param q_name: name of the qubit in dict_cfg
    :param n_walkers: number of walker points
    :param steps: number of steps
    :param thresh: threshold for evolution (between 0 and 1)
    :param evol: evolution rate (between 0 and 1)
    :param bounds: (n,2) list with the initial boundaries given in the order [pulse length (us), ramp length (us),
        frequency (MHz), gain (a.u.)], each element should be [lower bound, upper bound]
    :param n_points: number of datapoints for visibility to be calculated over
    :param dc: dc voltage [V]
    :param plot: if True, plots the data
    :param save: if True, saves the data
    :param collect: if True, returns the data
    """
    if q_name==None or n_walkers==None or steps==None or thresh==None or evol==None or bounds==None or n_points==None or dc==None:
        raise Exception("You must specify q_name, n_walkers, steps, thresh, evol, bounds, n_points, and dc")
    
    n_features=len(bounds)
    
    xfinal=[]
    for i in range(n_walkers):
        x=dict()
        for key in bounds.keys():
            x[key]=(bounds[key][1]-bounds[key][0])*np.random.random()+bounds[key][0]
        xfinal.append(x)
    
    x=copy.deepcopy(xfinal)

    xvals=[calc_visibility(q_name, n_points, dc, feats=xfinal[i]) for i in range(n_walkers)]
    xfinal=[xfinal]
    xvalsfinal=np.copy(xvals)
    indexes = np.arange(n_walkers)
    
    for i in tqdm(range(steps)):
        
        mask = np.random.rand(n_walkers, n_features)<thresh
        rs = np.random.randint(n_features, size=n_walkers)
        rs = np.tile(np.arange(n_features), n_walkers).reshape(n_walkers, n_features)==np.array(rs).reshape(n_walkers,1)
        mask = np.logical_or(mask, rs)

        for ii in tqdm(range(n_walkers)):
            np.random.shuffle(indexes)
            r1, r2, r3 = indexes[indexes != ii][0:3]
            
            y=dict()
            for j, key in enumerate(bounds.keys()):
                y[key] = x[r1][key]+evol*(x[r2][key]-x[r3][key])
                y[key] = y[key]*mask[ii][j]+x[ii][key]*(1-mask[ii][j])
                if y[key]<0:
                    y[key]=x[ii][key]
            try:
                yval=calc_visibility(q_name, n_points, dc, feats=y)
            except:
                y=x[ii]
                yval=calc_visibility(q_name, n_points, dc, feats=y)

            if yval>xvals[ii]:
                x[ii]=y
                xvals[ii]=yval
        
        x=copy.deepcopy(x)
        xfinal.append(x)
        xvalsfinal=np.vstack((xvalsfinal, xvals))
    
    title="Differential Evolution"

    if plot:
        fig, axs = plt.subplots(n_features+1,1, figsize=(10,5*(n_features+1)))
        for i in range(steps+1):
            for j, key in enumerate(bounds.keys()):
                plotvals=np.array([sub[key] for sub in xfinal[i]])
                axs[j].scatter(np.ones(n_walkers)*i, plotvals)
                axs[j].set_xlabel('step')
                axs[j].set_ylabel(key)
            axs[n_features].scatter(np.ones(n_walkers)*i, xvalsfinal[i])
            axs[n_features].set_xlabel('step')
            axs[n_features].set_ylabel('visibilities')
        
        if plot and save:
            plt.savefig('images/{}/Stark-{}-{}-{}.pdf'.format(q_name, title, date.today(), trial))
        
    if save:
        if trial is None:
            trial='01'

        #SAVE THE DATA HERE
        #SAVE THE DATA HERE
        #SAVE THE DATA HERE

        config = {
            "n_walkers": n_walkers,
            "n_steps": steps,
            "threshold": thres,
            "evol_rate": evol,
            "bounds":bounds,
            "n_points": n_points,
            "dc": dc
        }
        dumptopickle('./measurement_data/{}/Settings-Stark-{}-{}-{}.pickle'.format(q_name, title, date.today(), trial), config)

    if collect==True:
        return xfinal, xvalsfinal

class T1Program(RAveragerProgram):
    def initialize(self):
        cfg=self.cfg
        self.q_rp=self.ch_page(cfg["qubit_ch"])     # get register page for qubit_ch
        self.r_wait = 3
        self.regwi(self.q_rp, self.r_wait, cfg["start"])

        self.declare_gen(ch=cfg["qubit_ch"], nqz=2)
        self.declare_gen(ch=cfg["res_ch"], nqz=2)
        self.declare_gen(ch=cfg["dc_ch"], nqz=1)
        
        for ch in [0]:
            self.declare_readout(ch=ch, length=cfg["readout_length"],
                                 freq=cfg["f_res"], gen_ch=cfg["res_ch"])
        
        global phase
        phase=self.deg2reg(cfg["phase"])
        self.add_DRAG(ch=cfg["qubit_ch"], name="drive", length=cfg["pulse_length"], sigma=cfg["pulse_length"]/4,
                      delta=cfg["delta"], alpha=cfg["alpha"])
        self.add_gauss(ch=cfg["res_ch"], name="measure", sigma=cfg["res_sigma"]/4, length=cfg["res_sigma"])
        #setting up DC - must be done in this cell to be updated
        drivelen = (cfg["ramp_delay"] + cfg["probe_length"])*16 #add ramp delay so we can ramp up correctly
        rolen = (cfg["drive_ro_delay"] + cfg["res_length"] + 50)*16 #50 clock cycles to account for trigger setting
        
        #DC waveform (running in background for whole pulse sequence)
        waveform = np.hstack((np.ones(drivelen)*cfg["topamp"], np.ones(rolen)*cfg["baseamp"]))
        self.synci(200)
        
    def body(self):
        cfg=self.cfg
        
        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=4, mode="periodic")

        self.pulse(ch=cfg["dc_ch"])
        
        f_res=self.freq2reg(cfg["f_res"], gen_ch=cfg["res_ch"], ro_ch=0)
        f_ge=self.freq2reg(cfg["f_ge"], gen_ch=cfg["qubit_ch"])
        
        self.set_pulse_registers(ch=cfg["qubit_ch"], style="arb", freq=f_ge, phase=0, gain=cfg["pi_gain"],
                                 waveform="drive")
        
        self.pulse(ch=self.cfg["qubit_ch"])  #play probe pulse
        self.sync(self.q_rp,self.r_wait)
        
        self.set_pulse_registers(ch=cfg["res_ch"], style="flat_top", waveform="measure", freq=f_res, length=cfg["res_length"], 
                                 gain=cfg["res_gain"], phase=phase)

        #trigger measurement, play measurement pulse, wait for qubit to relax
        self.measure(pulse_ch=self.cfg["res_ch"], 
             adcs=[0],
             adc_trig_offset=self.cfg["adc_trig_offset"],
             wait=True,
             syncdelay=self.us2cycles(self.cfg["relax_delay"]))
    
    def update(self):
        self.mathi(self.q_rp, self.r_wait, self.r_wait, '+', self.us2cycles(self.cfg["step"])) # update frequency list index

def T1(q_name=None, delay_tuple=None, dc_bias=0.0, zpa=0.0,  zpa_tuple=None, reps=300,
                path="Jeffrey_Q_B.dir/", title=None, collect=None, fit=False):
    
    """
    Function to find T1
    
    :param q_name: name of the qubit in dict_cfg
    :param start: starting delay [us]
    :param stop: maximum delay [us]
    :param expts: delay points (int)
    :param reps: # of reps to be averaged over
    :param dc: dc voltage [V]
    :params plot, save, trial: see plotsave1d function above
    :param collect: if True, returns data. Returns T1 if False or None

    """
    if q_name==None or delay_tuple==None or dc_bias==None or (zpa==None and zpa_tuple==None):
        raise Exception("You must specify q_name, delay_tuple, dc_bias and zpa")

    if zpa_tuple!=None:
        zpa_start = zpa_tuple[0] ; zpa_stop = zpa_tuple[1] ; zpa_expts = zpa_tuple[2]
        yvar = np.linspace(zpa_start, zpa_stop, zpa_expts)
        dim = 2
    else:
        yvar = np.array(zpa)
        dim = 1

    start = soccfg.us2cycles(delay_tuple[0])
    stop = soccfg.us2cycles(delay_tuple[1])
    expts = delay_tuple[2]

    step=(stop-start)/expts
    expt_cfg={ "start":start, "step":step, "expts":expts, "reps":reps, "baseamp":dc_bias,
            "relax_delay":250
           }
    dict_cfg = loadfrompickle(q_name)
    config={**dict_cfg, **expt_cfg}

    if config["threshold"] is not None:
        corr=config["readout_length"]
    else:
        corr=1
    
    results=[]
    for i in yvar:
        config["topamp"] = dc2gain(dict_cfg["dc_ch"], i)
        t1p=T1Program(soccfg, config)
        x_pts, avgi, avgq = t1p.acquire(soc, threshold=config["threshold"], load_pulses=True, progress=True, debug=False)
        results.append(np.sqrt(avgi[0][0]**2+avgq[0][0]**2)*corr)

    if fit:
        try:
            p_guess=[results[0]-results[-1],results[-1],10]

            p_optt, p_covt = curve_fit(exp_decay, x_pts, results, p0 = p_guess)
            p_errt = np.sqrt(np.diag(p_covt))

            fitfunc=exp_decay(x_pts,p_optt[0],p_optt[1],p_optt[2])
            T1=p_optt[2]
        except:
            T1=0
            print("Fit did not converge")
    else:
        fitfunc=None
    

    if title==None:
            title = "T1 Experiment"
            
        #plotting using labrad format
    if dim==2:
        delay_dict = {'drange':(x_pts*1000, "Delay", "ns")} ; g_dict = {'grange':(yvar, "DC Voltage", "V")}
        amplitudes_tuple = (zampl, "Amplitude", "DAC Units", "Amplitude") ; angles_tuple = (zangl, "Angle", "Radians", "Angle")
        labrad2d(q_name=q_name, path=save_path, title=title, config=dict_cfg, xdict=delay_dict, ydict=g_dict, amplitudes=amplitudes_tuple, angles=angles_tuple)
    
    if dim==1:
        delay_dict = {'drange':(x_pts*1000, "Delay", "ns")}
        amplitudes_tuple = (zampl, "Amplitude", "DAC Units", "Amplitude") ; angles_tuple = (zangl, "Angle", "Radians", "Angle")
        labrad1d(q_name=q_name, path=save_path, title=title, config=dict_cfg, xdict=delay_dict, amplitudes=amplitudes_tuple, angles=angles_tuple)


    plotsave1d(plot=plot, q_name=q_name, save=save, title="T1", trial=trial, xvar=np.linspace(start,start+step*expts,expts),
               xvarname="Delay (us)", ydata=results, ydataname="Qubit population", fitfunc=fitfunc, config=config)
    
    if collect:
        return results
    else:
        if fit:
            return T1

class RamseyProgram(RAveragerProgram):
    def initialize(self):
        cfg=self.cfg
        
        self.q_rp=self.ch_page(cfg["qubit_ch"])     # get register page for qubit_ch
        self.r_wait = 3
        self.r_phase2 = 5
        self.r_phase=self.sreg(cfg["qubit_ch"], "phase")
        self.regwi(self.q_rp, self.r_wait, cfg["start"])
        self.regwi(self.q_rp, self.r_phase2, 0)
        
        self.declare_gen(ch=cfg["qubit_ch"], nqz=2) #Qubit
        self.declare_gen(ch=cfg["res_ch"], nqz=2)
        
        for ch in [0,1]:
            self.declare_readout(ch=ch, length=cfg["readout_length"],
                                 freq=cfg["f_res"], gen_ch=cfg["res_ch"])
        
        global phase
        phase=self.deg2reg(cfg["phase"])
        self.add_DRAG(ch=cfg["qubit_ch"], name="drive", length=cfg["pulse_length"], sigma=cfg["pulse_length"]/4,
                      delta=cfg["delta"], alpha=cfg["alpha"])
        self.add_gauss(ch=cfg["res_ch"], name="measure", sigma=cfg["res_sigma"]/4, length=cfg["res_sigma"])
        
        self.synci(200)
        
    def body(self):
        cfg=self.cfg
        
        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=4, mode="periodic")

        self.pulse(ch=cfg["dc_ch"])
        
        f_res=self.freq2reg(cfg["f_res"], gen_ch=cfg["res_ch"], ro_ch=0) # conver f_res to dac register value
        f_ge=self.freq2reg(cfg["f_ge"], gen_ch=cfg["qubit_ch"])

        self.set_pulse_registers(ch=cfg["qubit_ch"], style="arb", freq=f_ge, phase=0, gain=int(cfg["pi_gain"]/2), 
                                 waveform="drive")
        
        self.regwi(self.q_rp, self.r_phase, 0)
        
        self.pulse(ch=self.cfg["qubit_ch"])  #play probe pulse
        self.mathi(self.q_rp, self.r_phase, self.r_phase2,"+",0)
        self.sync(self.q_rp,self.r_wait)

        self.pulse(ch=self.cfg["qubit_ch"])  #play probe pulse
        self.sync_all(cfg["drive_ro_delay"])

        self.pulse(ch=cfg["dc_ch"])

        self.set_pulse_registers(ch=cfg["res_ch"], style="flat_top", waveform="measure", freq=f_res, length=cfg["res_length"], 
                                 gain=cfg["res_gain"], phase=phase)
        
        self.trigger(adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=cfg["adc_trig_offset"])
        
        self.pulse(ch=cfg["res_ch"])
        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))

        
    def update(self):
        self.mathi(self.q_rp, self.r_wait, self.r_wait, '+', self.cfg["step"])
        self.mathi(self.q_rp, self.r_phase2, self.r_phase2, '+', self.cfg["phase_step"])

def T2(q_name=None, detuning=None, start=None, stop=None, p_step=None, expts=None, reps=None, dc=None, plot=None, save=None, trial=None, collect=None, fit=None):
    
    """
    Ramsey T2 Experiment
    
    :param q_name: name of the qubit in dict_cfg
    :param detuning: detuning from drive frequency [MHz]
    :param start: starting delay [us]
    :param stop: maximum delay [us]
    :param p_step: phase step [degrees]
    :param expts: delay points (int)
    :param reps: # of reps to be averaged over (int)
    :param dc: dc voltage [V]
    :param plot: if True, plots the data
    :param save: if True, saves the data
    :param collect: if True, returns data. Returns T2 and actual detuning if False or None.
    
    """

    if q_name==None or detuning==None or start==None or stop==None or p_step==None or expts==None or reps==None or dc==None:
        raise Exception("You must specify q_name, detuning, start, stop, p_steps, expts, reps, and dc")
    
    step=(stop-start)/expts
    expt_cfg={"start":soccfg.us2cycles(start), "step":soccfg.us2cycles(step), "phase_step": soccfg.deg2reg(p_step, gen_ch=2),
              "expts":expts, "reps": reps, "rounds": 1, "voltage":dc
           }
    dict_cfg = loadfrompickle(q_name)
    config={**dict_cfg, **expt_cfg}

    config["f_ge"]-=detuning
    if config["threshold"] is not None:
        corr=config["readout_length"]
    else:
        corr=1
    t2p=RamseyProgram(soccfg, config)
    x_pts, avgi, avgq= t2p.acquire(soc,threshold=config["threshold"], load_pulses=True,progress=True, debug=False)
    x_pts = soccfg.cycles2us(x_pts)
    results=np.sqrt(avgi[0][0]**2+avgq[0][0]**2)*corr
    
    if fit:
        try:
            loc_pt1 = np.argmax(results)
            loc_pt2 = np.argmin(results)
            
            A = np.max(results)-np.mean(results)
            W=np.pi/(x_pts[loc_pt1]-x_pts[loc_pt2])
            B=np.mean(results)
            T=-(x_pts[loc_pt1]-x_pts[loc_pt2])/np.log((np.max(results[2*loc_pt1-loc_pt2:])-np.mean(results))/
                                                      (np.max(results)-np.mean(results)))
            
            p_guess=[A, W, T, B, 0]
            p_optt2, p_covt2 = curve_fit(t2fit, x_pts, results, p0 = p_guess)
            p_errt2 = np.sqrt(np.diag(p_covt2))
            fitfunc=t2fit(x_pts,p_optt2[0],p_optt2[1],p_optt2[2],p_optt2[3],p_optt2[4])
            Delta=p_optt2[1]/(2*np.pi)
            info={'T2':p_optt2[2], 'Detuning':Delta}
        except:
            info=0
            print('Fit did not converge')
    else:
        fitfunc=None
    
    plotsave1d(plot=plot, save=save, q_name=q_name, title="T2", trial=trial, xvar=x_pts,
               xvarname="Delay (us)", ydata=results, ydataname="Qubit population", fitfunc=fitfunc, config=config)
    
    if collect:
        return results
    else:
        if fit:
            return info

class RamseyModProgram(RAveragerProgram):
    def initialize(self):
        cfg=self.cfg
        
        self.q_rp=self.ch_page(cfg["qubit_ch"])     # get register page for qubit_ch
        self.q_rp=self.ch_page(cfg["res_ch"])
        self.r_wait = 3
        self.r_wait2 = 4
        self.r_phase2 = 5
        self.r_phase=self.sreg(cfg["qubit_ch"], "phase")
        self.regwi(self.q_rp, self.r_wait, cfg["start"])
        self.regwi(self.q_rp, self.r_wait2, cfg["drive_ro_delay"])
        self.regwi(self.q_rp, self.r_phase2, 0)
        
        self.declare_gen(ch=cfg["qubit_ch"], nqz=2) #Qubit
        
        for ch in [0,1]:
            self.declare_readout(ch=ch, length=cfg["readout_length"],
                                 freq=cfg["f_res"], gen_ch=cfg["res_ch"])
        
        global phase
        phase=self.deg2reg(cfg["phase"])
        self.add_DRAG(ch=cfg["qubit_ch"], name="drive", length=cfg["pulse_length"], sigma=cfg["pulse_length"]/4,
                      delta=cfg["delta"], alpha=cfg["alpha"])
        self.add_gauss(ch=cfg["res_ch"], name="measure", sigma=cfg["res_sigma"]/4, length=cfg["res_sigma"])
        self.add_pulse(ch=cfg["dc_ch"], name="dc", idata=gen_dc_waveform(gen=cfg["dc_ch"], b_volt=cfg["voltage"],
                                                                       m_volt=cfg["mod_voltage"], freq=cfg["f_mod"]))
        self.synci(200)
        
    def body(self):
        cfg=self.cfg
        
        f_res=self.freq2reg(cfg["f_res"], gen_ch=cfg["qubit_ch"], ro_ch=0)
        f_ge=self.freq2reg(cfg["f_ge"], gen_ch=cfg["qubit_ch"])
        
        self.set_pulse_registers(ch=cfg["dc_ch"], outsel="input", style="arb", phase=0, freq=0, gain=32767,
                                 waveform="dc", mode="periodic")

        self.pulse(ch=cfg["dc_ch"])

        self.set_pulse_registers(ch=cfg["qubit_ch"], style="arb", freq=f_ge, phase=0, gain=int(cfg["pi_gain"]/2), 
                                 waveform="drive")
        
        self.regwi(self.q_rp, self.r_phase, 0)
        
        self.pulse(ch=self.cfg["qubit_ch"])  #play probe pulse
        self.mathi(self.q_rp, self.r_phase, self.r_phase2,"+",0)
        self.sync(self.q_rp,self.r_wait)

        self.pulse(ch=self.cfg["qubit_ch"])
        
        self.sync_all(cfg["drive_ro_delay"])

        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=4, mode="periodic")

        self.pulse(ch=cfg["dc_ch"])

        self.set_pulse_registers(ch=cfg["res_ch"], style="flat_top", waveform="measure", freq=f_res, length=cfg["res_length"], 
                                 gain=cfg["res_gain"], phase=phase)
        
        self.trigger(adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=cfg["adc_trig_offset"])
        
        self.pulse(ch=cfg["res_ch"])
        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))

        
    def update(self):
        self.mathi(self.q_rp, self.r_wait, self.r_wait, '+', self.cfg["step"]) # update the time between two π/2 pulses
        self.mathi(self.q_rp, self.r_phase2, self.r_phase2, '+', self.cfg["phase_step"]) # advance the phase of the LO for the second π/2 pulse

def T2Mod(q_name=None, detuning=None, start=None, stop=None, p_step=None, expts=None, reps=None, dc=None, dc_mod=None, 
          f_mod=None, plot=None, save=None, trial=None, collect=None, fit=None):
    
    """
    Ramsey T2 Experiment
    
    :param q_name: name of the qubit in dict_cfg
    :param detuning: detuning from drive frequency [MHz]
    :param start: starting delay [us]
    :param stop: maximum delay [us]
    :param p_step: phase step [degrees]
    :param expts: delay points (int)
    :param reps: # of reps to be averaged over (int)
    :param dc: dc voltage [V]
    :param plot: if True, plots the data
    :param save: if True, saves the data
    :param collect: if True, returns data. Returns T2 and actual detuning if False or None.
    
    """

    if q_name==None or detuning==None or start==None or stop==None or p_step==None or expts==None or reps==None or dc==None:
        raise Exception("You must specify q_name, detuning, start, stop, p_steps, expts, reps, and dc")
    
    step=(stop-start)/expts
    expt_cfg={"start":soccfg.us2cycles(start), "step":soccfg.us2cycles(step), "phase_step": soccfg.deg2reg(p_step, gen_ch=2),
              "expts":expts, "reps": reps, "rounds": 1, "voltage":dc, "mod_voltage":dc_mod, "f_mod":f_mod
           }
    dict_cfg = loadfrompickle(q_name)
    config={**dict_cfg, **expt_cfg}

    config["f_ge"]-=detuning
    if config["threshold"] is not None:
        corr=config["readout_length"]
    else:
        corr=1
    t2p=RamseyModProgram(soccfg, config)
    x_pts, avgi, avgq= t2p.acquire(soc,threshold=config["threshold"], load_pulses=True,progress=True, debug=False)
    x_pts = soccfg.cycles2us(x_pts)
    results=np.sqrt(avgi[0][0]**2+avgq[0][0]**2)*corr
    
    if fit:
        try:
            loc_pt1 = np.argmax(results)
            loc_pt2 = np.argmin(results)
            
            A = np.max(results)-np.mean(results)
            W=np.pi/(x_pts[loc_pt1]-x_pts[loc_pt2])
            B=np.mean(results)
            T=-(x_pts[loc_pt1]-x_pts[loc_pt2])/np.log((np.max(results[2*loc_pt1-loc_pt2:])-np.mean(results))/
                                                      (np.max(results)-np.mean(results)))
            
            p_guess=[A, W, T, B, 0]
            p_optt2, p_covt2 = curve_fit(t2fit, x_pts, results, p0 = p_guess)
            p_errt2 = np.sqrt(np.diag(p_covt2))
            fitfunc=t2fit(x_pts,p_optt2[0],p_optt2[1],p_optt2[2],p_optt2[3],p_optt2[4])
            Delta=p_optt2[1]/(2*np.pi)
            info={'T2':p_optt2[2], 'Detuning':Delta}
        except:
            info=0
            print('Fit did not converge')
    else:
        fitfunc=None
    
    plotsave1d(plot=plot, save=save, q_name=q_name, title="T2", trial=trial, xvar=x_pts,
               xvarname="Delay (us)", ydata=results, ydataname="Qubit population", fitfunc=fitfunc, config=config)
    
    if collect:
        return results
    else:
        if fit:
            return info

class summon_gate(AveragerProgram):
    """Function to summon a specified sequence of gates. Requires tune-up before using

    Parameters (all fed in through AveragerProgram):
    ["gate_sequence"] (int array): series of gates [0-8] and pauses [9] 
    ["detuning"] (int): amount of detuning in MHz
    ["sync_time"] (float): wait time between gates in clock cycles
    ["wait_sequence"] (float array): series of floats, each of which corresponds to the length of the pause in ["gate_sequence"]
    """
    
    def initialize(self):
        cfg=self.cfg
        
        self.declare_gen(ch=cfg["qubit_ch"], nqz=2) #Qubit
        self.declare_gen(ch=cfg["res_ch"], nqz=2)
        for ch in [0]:
            self.declare_readout(ch=ch, length=cfg["readout_length"],
                                 freq=cfg["f_res"], gen_ch=cfg["res_ch"])
        
        global phase
        phase=self.deg2reg(0, gen_ch=cfg["qubit_ch"])
        global negphase
        negphase=self.deg2reg(180, gen_ch=cfg["qubit_ch"])
        global yphase
        yphase=self.deg2reg(90, gen_ch=cfg["qubit_ch"])
        global negyphase
        negyphase=self.deg2reg(270, gen_ch=cfg["qubit_ch"])
        global resphase
        resphase=self.deg2reg(cfg["phase"])
        
        global f_res
        global f_ge

        f_res=self.freq2reg(cfg["f_res"], gen_ch=cfg["res_ch"], ro_ch=0)
        f_ge=self.freq2reg(cfg["f_ge"]-cfg["detuning"], gen_ch=cfg["qubit_ch"])
            
        self.add_DRAG(ch=self.cfg["qubit_ch"], name="drive", length=self.cfg["pulse_length"], 
                      sigma=self.cfg["pulse_length"]/4, delta=self.cfg["delta"], alpha=self.cfg["alpha"])
        self.add_gauss(ch=self.cfg["res_ch"], name="measure", sigma=self.cfg["res_sigma"]/4,
                       length=self.cfg["res_sigma"])
        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=4, mode="periodic")
        
        self.synci(200)
        
    def body(self):
        cfg=self.cfg
        
        w_iter = 0
        for i in self.cfg["gate_sequence"]:    
            if i == 9: #Pause in pulses
                self.sync_all(cfg["wait_sequence"][w_iter])
                w_iter += 1
            elif i == 0: #Identity gate
                self.pulse(ch=cfg["dc_ch"])
                self.set_pulse_registers(ch=self.cfg["qubit_ch"], freq=f_ge, gain=0, style="arb",waveform="drive", phase=phase)
                self.pulse(ch=self.cfg["qubit_ch"])
                self.sync_all(cfg["sync_time"])
            elif i == 1: #X
                self.pulse(ch=cfg["dc_ch"])
                self.set_pulse_registers(ch=self.cfg["qubit_ch"], freq=f_ge, gain=self.cfg["pi_gain"], style="arb", 
                                         waveform="drive", phase=phase)
                self.pulse(ch=self.cfg["qubit_ch"])
                self.sync_all(cfg["sync_time"])
            elif i == 2: #Y
                self.pulse(ch=cfg["dc_ch"])
                self.set_pulse_registers(ch=self.cfg["qubit_ch"], freq=f_ge, phase=yphase, gain=self.cfg["pi_gain"], 
                                         style="arb", waveform="drive")
                self.pulse(ch=self.cfg["qubit_ch"])
                self.sync_all(cfg["sync_time"])
            elif i == 3: #X2
                self.pulse(ch=cfg["dc_ch"])
                self.set_pulse_registers(ch=self.cfg["qubit_ch"], freq=f_ge, gain=self.cfg["pi2_gain"], 
                                         style="arb", waveform="drive", phase=phase)
                self.pulse(ch=self.cfg["qubit_ch"])
                self.sync_all(cfg["sync_time"])
            elif i == 4: #Y2
                self.pulse(ch=cfg["dc_ch"])
                self.set_pulse_registers(ch=self.cfg["qubit_ch"], freq=f_ge, phase=yphase,
                                     gain=self.cfg["pi2_gain"], style="arb", waveform="drive")
                self.pulse(ch=self.cfg["qubit_ch"])
                self.sync_all(cfg["sync_time"])
            elif i == 5: #-X2
                self.pulse(ch=cfg["dc_ch"])
                self.set_pulse_registers(ch=self.cfg["qubit_ch"], freq=f_ge, phase=negphase, gain=self.cfg["pi2_gain"],
                                         style="arb", waveform="drive")
                self.pulse(ch=self.cfg["qubit_ch"])
                self.sync_all(cfg["sync_time"])
            elif i == 6: #-Y2
                self.pulse(ch=cfg["dc_ch"])
                self.set_pulse_registers(ch=self.cfg["qubit_ch"], freq=f_ge, phase=negyphase,
                                     gain=self.cfg["pi2_gain"], style="arb", waveform="drive")
                self.pulse(ch=self.cfg["qubit_ch"])
                self.sync_all(cfg["sync_time"])
            elif i == 7: #-X
                self.pulse(ch=cfg["dc_ch"])
                self.set_pulse_registers(ch=self.cfg["qubit_ch"], freq=f_ge, gain=self.cfg["pi_gain"], phase=negphase, 
                                         style="arb", waveform="drive")
                self.pulse(ch=self.cfg["qubit_ch"])
                self.sync_all(cfg["sync_time"])
            elif i == 8: #-Y
                self.pulse(ch=cfg["dc_ch"])
                self.set_pulse_registers(ch=self.cfg["qubit_ch"], freq=f_ge, phase=negyphase, 
                                         gain=self.cfg["pi_gain"], style="arb", waveform="drive")
                self.pulse(ch=self.cfg["qubit_ch"])
                self.sync_all(cfg["sync_time"])
            else:
                raise ValueError("'cfg[gate_sequence] value does not correspond to basic gate generator [0,8] or a pause 9 '")
        #Readout
        self.sync_all(cfg["drive_ro_delay"])

        self.pulse(ch=cfg["dc_ch"])

        self.set_pulse_registers(ch=self.cfg["res_ch"], style="flat_top", waveform="measure", freq=f_res,
                                 length=self.cfg["res_length"], gain=self.cfg["res_gain"], phase=resphase)

        self.trigger(adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=self.cfg["adc_trig_offset"])

        self.pulse(ch=self.cfg["res_ch"])
        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))

def gen_Clifford_seq(m, gate=None):
    """Function to generate array of gates for single-qubit gate randomized Clifford benchmarking

    Parameters:
    m (int): number of Cliffords
    gate (int): number corresponding to the interleaved gate. If None, then no interleaved gate
    """
    
    gates = [0,1,2,3,4,5,6,7,8]
    gates_inv = [0,7,8,5,6,3,4,1,2]
    cliffords = [[0],[1],[2],[2,1],[3,4],[3,6],[5,4],[5,6],[4,3],[4,5],[6,3],[6,5],
                 [3],[5],[4],[6],[5,4,3],[5,6,3],[1,4],[1,6],[2,3],[2,5],[3,4,3],[5,4,5]]
    
    gate_arr = [np.diag([1,1]), #I
                np.array([[0,-1j], [-1j,0]]), #X
                np.array([[0,-1], [1,0]]), #Y
                np.array([[1,-1j], [-1j,1]])/np.sqrt(2), #X2
                np.array([[1,-1], [1,1]])/np.sqrt(2), #Y2
                np.array([[1,1j], [1j,1]])/np.sqrt(2), #-X2
                np.array([[1,1], [-1,1]])/np.sqrt(2), #-Y2
                np.array([[0,1j], [1j,0]]), #-X
                np.array([[0,1], [-1,0]])] #-Y
    
    
    a = np.random.randint(0,23,int(m))
    gate_seq = []
    op_seq = gate_arr[0]
    
    for i in a:
        gate_seq.extend(cliffords[i])
        if gate is not None:
            gate_seq.append(gate)
    for j in gate_seq:
        op_seq = np.dot(op_seq, gate_arr[j])
    for g_try in range(24):
        checking_gate_seq = cliffords[g_try]
        check_gate = gate_arr[0]
        for k in checking_gate_seq:
            check_gate = np.dot(check_gate, gate_arr[k])
        gate_compare = np.dot(op_seq, check_gate)

        if np.abs(gate_compare[0,1])<0.01:
            gate_seq.extend(cliffords[g_try])
            break
            
    return gate_seq

def SingleGateRandFidelity(q_name=None, m_start=None, m_stop=None, m_expts=None, gate=None, k=None, reps=None, dc=None, plot=None, save=None, trial=None, collect=None):
    """
    Single Qubit Interleaved Clifford Fidelity
    
    :param q_name: name of qubit in dict_cfg (str)
    :param m_arr: series that specifies how many Cliffords are executed per trial [int array]
    :param gate: identity of the interleaved gate [int from 0,8]
    :param k: how many different sequences are generated each trial [int]
    :param reps: number of repetitions used in the sg.acquire function [int]
    :param plot: if True, plots the data
    :param save: if True, saves the data
    :param collect: if True, returns the data
    """

    if q_name==None or m_start==None or m_stop==None or m_expts==None or gate==None or k==None or reps==None or dc==None:
        raise Exception("You must specify q_name, m_start, m_stop, m_expts, gate, k, reps, and dc")
    
    m_arr = np.linspace(m_start, m_stop, m_expts)

    dict_cfg = loadfrompickle(q_name)
    if dict_cfg["threshold"] is not None:
        corr=dict_cfg["readout_length"]
    else:
        corr=1
    probs=[]   
    for m in tqdm(m_arr):
        prob=[]
        for ii in range(k):
            gates = gen_Clifford_seq(m)
            cliff_cfg={"reps":reps, "gate_sequence": gates, "wait_sequence": 0, 
                       "detuning":0, "sync_time":soccfg.us2cycles(0.01, gen_ch=2), "voltage":dc}
            config={**dict_cfg, **cliff_cfg}
            sg=summon_gate(soccfg, config)
            avgi, avgq =sg.acquire(soc, threshold=config["threshold"], load_pulses=True, progress=False,debug=False)
            prob.append(np.sqrt(avgi[0][0]**2+avgq[0][0]**2)*corr)
        probs.append(np.average(prob))
        
    
    gprobs=[]
    for m in tqdm(m_arr):
        for ii in range(k):
            gprob=[]
            ggates = gen_Clifford_seq(m, gate)
            config["gate_sequence"]=ggates
            sg=summon_gate(soccfg, config)
            avgi, avgq=sg.acquire(soc, threshold=config["threshold"], load_pulses=True, progress=False,debug=False)
            gprob.append(np.sqrt(avgi[0][0]**2+avgq[0][0]**2)*corr)
        gprobs.append(np.average(gprob))
    
    p_guess = [-0.5,0.9,0.5]
    p_opt, p_cov = curve_fit(p_decay, m_arr*1.875, probs, p0 = p_guess, bounds=((-0.7, 0.1, 0.3), (-0.3, 1, 0.7)))

    gp_opt, gp_cov = curve_fit(p_decay, m_arr*1.875, gprobs, p0 = p_guess, bounds=((-0.7, 0.1, 0.3), (-0.3, 1, 0.7)))

    p = p_opt[1]
    p_inter = gp_opt[1]
    rand_fidelity = 1 - ((1-p)/(2*1.875))
    inter_fidelity = (1+p_inter/p)/2

    xfit = m_arr*1.875
    yfit = p_decay(xfit, p_opt[0],p_opt[1],p_opt[2])
    
    gyfit = p_decay(xfit, gp_opt[0],gp_opt[1],gp_opt[2])
    
    gate_strings = ['I', 'X', 'Y' , 'X2', 'Y2', '-X2', '-Y2', '-X', '-Y']
    
    plotsave1d(plot=plot, save=save,  q_name=q_name, title="Randomized Clifford Gates Benchmarking Fidelity", 
               trial=trial, xvar=xfit, xvarname="Number of Clifford Gates (m)", ydata=probs, ydataname="Qubit Population", 
               fitfunc=yfit, config=config)
    
    plotsave1d(plot=plot, save=save,  q_name=q_name, title=str(gate_strings[int(gate)]) + " Gate Interleaved Randomized Clifford Benchmarking Fidelity", 
               trial=trial, xvar=xfit, xvarname="Number of Clifford Gates (m)", ydata=gprobs, ydataname="Qubit Population", 
               fitfunc=gyfit, config=config)

    soc.reset_gens()

    if collect:
        return np.vstack([probs, gprobs]).T
    else:
        return rand_fidelity, inter_fidelity
    
def HahnEcho(q_name=None, tau_start=None, tau_stop=None, tau_expts=None, reps=None, dc=None, detuning=None, plot=None, save=None, collect=None, trial=None, fit=None):
    """
    Hahn Echo

    :param q_name: name of qubit in dict_cfg [str]
    :param tau_arr: array of the times used for individual Hahn echo experiments [np.arr in us]
    :param reps: number of repetitions used in the sg.acquire() function [int]
    :param detuning: detuning of the qubit [MHz]
    :param plot: if True, plots the data
    :param save: if True, saves the data
    :param collect: if True, returns the data
    """
    
    if q_name==None or tau_start==None or tau_stop==None or tau_expts==None or reps==None or dc==None:
        raise Exception("You must specify q_name, tau_arr, reps, and dc")

    tau_arr = np.linspace(tau_start, tau_stop, tau_expts)

    if detuning==None:
        detuning=0

    probs = []
    gates = [3,9,1,9,3] #Hahn sequence: X2, I, X, I, X2
    dict_cfg = loadfrompickle(q_name)
    expt_cfg={"reps":reps, "gate_sequence": gates, "detuning":detuning, 
                      "sync_time":soccfg.us2cycles(0.01, gen_ch=2), "voltage":dc}
    config={**dict_cfg, **expt_cfg}
    if config["threshold"] is not None:
        corr=config["readout_length"]
    else:
        corr=1
    for tau in tqdm(soccfg.us2cycles(tau_arr, gen_ch=2)):
            config["wait_sequence"]=[tau, tau]
            he=summon_gate(soccfg, config)
            avgi, avgq=he.acquire(soc, threshold=config["threshold"], load_pulses=True, 
                                  progress=False,debug=False)
            probs.append(np.sqrt(avgi[0][0]**2+avgq[0][0]**2)*corr)
    if fit:
        try:
            p_guess = [0.5,0.2]
            p_opt, p_cov = curve_fit(hahn_decay, tau_arr/2, probs, p0 = p_guess)
            p_err = np.sqrt(np.diag(p_cov))
            fitfunc = hahn_decay(tau_arr/2, p_opt[0], p_opt[1])
            T2=2*p_opt[1]
        except:
            T2=0
            print("Fit did not converge")
    else:
        fitfunc=None
    
    plotsave1d(plot=plot, save=save,  q_name=q_name, title="Hahn Echo", 
               trial=trial, xvar=tau_arr/2, xvarname="Wait time (us)", ydata=probs, ydataname="Qubit Population", 
               fitfunc=fitfunc, config=config)

    soc.reset_gens()

    if collect:
        return probs
    else:
        if fit:
            return T2

def CPMG(q_name=None, cpmg_start=None, cpmg_stop=None, cpmg_expts=None, reps=None, detuning=None, gate_start=None, gate_stop=None, gate_expts=None, dc=None, plot=None, save=None, trial=None, collect=None):
    """

    CPMG Experiment

    :param q_name: name of qubit in dict_cfg (str)
    :param cpmg_arr: array of the times used for individual Hahn echo experiments [np.arr in us]
    :param reps: number of repetitions used in the sg.acquire function [int]
    :param dc: dc voltage [V]
    :param plot: if True, plots the data
    :param save: if True, saves the data
    :param collect: if True, returns the data
    """
    
    if q_name==None or cpmg_start==None or cpmg_stop==None or cpmg_expts==None or reps==None or gate_start==None or gate_stop==None or gate_expts==None or dc==None:
        raise Exception("You must specify q_name, cpmg_start, cpmg_stop, cpmg_expts, reps, detuning, gate_start, gate_stop, gate_expts, and dc.")

    tau_arr = np.logspace(cpmg_start, cpmg_stop, cpmg_expts)
    gate_arr = np.linspace(gate_start, gate_stop, gate_expts)

    if detuning==None:
        detuning=0

    t2s = []
    t2_errs = []

    dict_cfg = loadfrompickle(q_name)
    expt_cfg={"reps":reps, "detuning":detuning, 
                      "sync_time":soccfg.us2cycles(0.01, gen_ch=2), "voltage":dc}
    config={**dict_cfg, **expt_cfg}
    if config["threshold"] is not None:
        corr=config["readout_length"]
    else:
        corr=1

    for n in tqdm(gate_arr):
        n = int(n)
        tau_probs = []
        gates = np.array([3] + [9,1]*n + [9,3])
        config["gates"]=gates
        for tau in tau_arr:
            wait_seq = np.array([soccfg.us2cycles(tau/(2*n))] + [soccfg.us2cycles(tau/n)]*(n-1) + [soccfg.us2cycles(tau/(2*n))])
            config["wait_sequence"]=wait_seq
            cpmg=summon_gate(soccfg, config)
            avgi, avgq=cpmg.acquire(soc, threshold=config["threshold"], load_pulses=True, 
                                  progress=False,debug=False)
            tau_probs.append(np.sqrt(avgi[0][0]**2+avgq[0][0]**2)*corr)
    
        #forcing curve to go through (0,0) with some tomfoolery
        temp_tau_arr = [0] + tau_arr
        temp_tau_probs = [0] + temp_tau_probs
        #weighting array
        prob_weights = np.empty(len(temp_tau_probs))
        prob_weights.fill(10) #10 is arb
        prob_weights[0] = 0.0001 #this ensures we go through (0,0)
        
        p_guess = [0,0,30]
        p_opt, p_cov = curve_fit(exp_decay, temp_tau_arr*tau_max, temp_tau_probs, p0 = p_guess, bounds=bounds)#NO BOUNDS SET
        p_err = np.sqrt(p_cov)
        t2s.append(p_opt[2])
        t2_errs.append(p_err[2])

        fig, ax = plt.subplots()
        t2string = "T2 = " + str(p_opt[2])
        ax.plot(tau_arr*tau_max, tau_probs, 'ko', label=t2string)
        xfit = np.linspace(0,tau_max,1000)
        yfit = exp_decay(xfit, p_opt[0], p_opt[1], p_opt[2])
        ax.plot(xfit, yfit,'-')
        plt.xlabel("Wait time (us)")
        plt.ylabel("Qubit population")
        plt.legend()
    

    plotsave1d(plot=plot, save=save,  q_name=q_name, title="CPMG", 
               trial=trial, xvar=gate_arr, xvarname="Wait time (us)", ydata=t2s, ydataname="T2 time", 
               fitfunc=None, config=config)

class FlatTopDCProgram(AveragerProgram):

    def initialize(self):
        cfg=self.cfg

        self.declare_gen(ch=cfg["qubit_ch"], nqz=2) #Qubit
        self.declare_gen(ch=cfg["res_ch"], nqz=2)

        for ch in [0,1]:
            self.declare_readout(ch=ch, length=cfg["readout_length"],
                                 freq=cfg["f_res"], gen_ch=cfg["res_ch"])
        
        global phase
        phase=self.deg2reg(cfg["phase"])

        self.add_gauss(ch=cfg["res_ch"], name="measure", sigma=cfg["res_sigma"]/4, length=cfg["res_sigma"])
        self.add_pulse(ch=cfg["dc_ch"], name="dc",idata=dc_gauss(mu=cfg["dc_length"]/2-0.5, si=cfg["dc_sigma"], 
                                                                 length=cfg["dc_length"], dc_bias=cfg["voltage"],
                                                                 dc_pulse=cfg["pulse_voltage"], dc_ch=cfg["dc_ch"]))
            
        self.synci(200)
        
    def body(self):
        cfg=self.cfg
        
        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=4, mode="periodic")
        
        self.pulse(ch=cfg["dc_ch"])
        
        self.set_pulse_registers(ch=cfg["dc_ch"], style="flat_top", phase=0, freq=0, gain=32767,
                                 length=cfg["pulse_length"], waveform="dc")
        
        self.pulse(ch=cfg["dc_ch"])
        
        
        self.set_pulse_registers(ch=cfg["res_ch"], style="flat_top", waveform="measure", freq=f_res, length=cfg["res_length"], 
                                 gain=cfg["res_gain"], phase=phase)
        
        self.trigger(adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=cfg["adc_trig_offset"])
        
        self.pulse(ch=cfg["res_ch"])
        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))

def FlatTopDC(dc_length=None, dc_sigma=None, reps=None, dc_bias=None, dc_pulse=None):

    expt_cfg={"dc_length":dc_length, "dc_sigma":dc_sigma, "reps":reps, "voltage":dc_bias, "pulse_voltage":dc_pulse}
    dict_cfg = loadfrompickle("Q_B")
    config={**dict_cfg, **expt_cfg}

    if config["threshold"] is not None:
        corr=config["readout_length"]
    else:
        corr=1
    t2p=FlatTopDCProgram(soccfg, config)
    avgi, avgq= t2p.acquire(soc,threshold=config["threshold"], load_pulses=True,progress=True, debug=False)
    soc.reset_gens()

class Test0ActiveResetProgram(AveragerProgram):   
    def initialize(self):
        cfg=self.cfg      
        
        self.regwi(0,1,0)
        self.regwi(0,7,0)
        
        self.r_thresh = 6
        self.regwi(0,self.r_thresh,int(cfg["threshold"]*cfg["readout_length"]))
#         self.regwi(0,self.r_thresh,int(cfg["threshold"]*cfg["readout_length"]))
#         self.memwi(0,self.r_thresh,127)
        self.q_rp=self.ch_page(self.cfg["qubit_ch"])
        self.r_gain=self.sreg(cfg["qubit_ch"], "gain")
        self.r_freq=self.sreg(cfg["qubit_ch"], "freq")
        self.r_phase = self.sreg(cfg["qubit_ch"], "phase")
        
        self.declare_gen(ch=cfg["res_ch"], nqz=2)
        self.declare_gen(ch=cfg["qubit_ch"], nqz=2)
        self.declare_readout(ch=0, length=cfg["readout_length"],
                                 freq=cfg["f_res"], gen_ch=cfg["res_ch"]) #freq in declare_readout must be in MHz
        
        global f_res
        global f_ge
        
        f_res=self.freq2reg(cfg["f_res"], gen_ch=cfg["qubit_ch"], ro_ch=0)
        f_ge=self.freq2reg(cfg["f_ge"], gen_ch=cfg["qubit_ch"])
        
        global phase
        phase=self.deg2reg(cfg["phase"])
        
        self.add_DRAG(ch=cfg["qubit_ch"], name="drive", length=cfg["pulse_length"], sigma=cfg["pulse_length"]/4,
                      delta=cfg["delta"], alpha=cfg["alpha"])
        self.add_gauss(ch=cfg["res_ch"], name="measure", sigma=cfg["res_sigma"]/4, length=cfg["res_sigma"])
        
        self.sync_all(self.us2cycles(500))
    
    def body(self):

        cfg=self.cfg
        
        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=4, mode="periodic")
        self.set_pulse_registers(ch=cfg["qubit_ch"], style="arb", waveform="drive", freq=f_ge, gain=cfg["qubit_gain"], phase=0)
        self.set_pulse_registers(ch=cfg["res_ch"], style="flat_top", waveform="measure", freq=f_res, length=cfg["res_length"], 
                                 gain=cfg["res_gain"], phase=phase)
        
        #Initial drive
        self.pulse(ch=cfg["dc_ch"])
        self.pulse(ch=cfg["qubit_ch"])
        self.sync_all(cfg["drive_ro_delay"])

        #Measure pulse
        self.pulse(ch=cfg["dc_ch"])
        self.trigger(adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=cfg["adc_trig_offset"])
        self.pulse(ch=cfg["res_ch"])
        self.wait_all(self.us2cycles(0.09))
        
        self.read(0,0,"lower",2)
        self.read(0,0,"upper",3)
        self.mathi(0,self.r_thresh,self.r_thresh,'+',10000)
        self.mathi(0,2,2,'+',10000)
        self.mathi(0,3,3,'+',10000)
        self.memwi(0,2,125) #### TESTING
        self.memwi(0,3,126)
        
        self.condj(0,2,'<',self.r_thresh,'after_reset')
        
        self.regwi(self.q_rp, self.r_gain, self.cfg["pi_gain"])  #set up pi pulse
        self.mathi(0,7,7,'+',1)
        self.pulse(ch=cfg["dc_ch"])
        self.pulse(ch=self.cfg["qubit_ch"], t=0)
        self.sync_all(cfg["drive_ro_delay"])
        
        self.sync_all(self.us2cycles(0.2))

        self.label('after_reset')
        
        self.sync_all(self.us2cycles(0.2))
        self.memwi(0,7,132)
        self.memwi(self.q_rp, self.r_gain,127)
            
        self.pulse(ch=cfg["dc_ch"])
        self.measure(pulse_ch=self.cfg["res_ch"], 
              adcs=[0],
              adc_trig_offset=self.cfg["adc_trig_offset"],
              wait=True,
              syncdelay=self.us2cycles(self.cfg["relax_delay"]))
        
        self.read(0,0,"lower",8)
        self.memwi(0,8,131)

def Test0ActiveReset(q_name, start, stop, expts, reps, dc, title, waveform=False, plot=None, verbose=False):
    
    expt_cfg={"start":start, "stop":stop, "expts":expts, "reps": reps, "voltage": dc
       }
    
    dict_cfg = loadfrompickle(q_name)
    config={**dict_cfg, **expt_cfg}     
    expt_pts=np.linspace(start, stop, expts).astype(int)
    pre_reset=[]
    post_reset=[]
    
    config["pi_gain"]=16000
    config["pulse_length"]=18
    config["f_ge"]=5005
    
    for i in tqdm(expt_pts):
        config["qubit_gain"]=i
        areset=Test0ActiveResetProgram(soccfg, config)
        
        if verbose == True:
            result = soc.tproc.single_read(addr=125)
            result1 = soc.tproc.single_read(addr=126)
            check = soc.tproc.single_read(addr=127)
            final = soc.tproc.single_read(addr=131)
            average = soc.tproc.single_read(addr=132)
            if final > 100000:
                final = final - 4294967296
            if result > 100000:
                result = result - 4294967296
            if result1 > 100000:
                result1 = result1 - 4294967296
            print("I,Q data = ", result, result1)
            print("Cond. Check = ", check, " number = ", average)
            print("Final I = ", final)

        
        if waveform==True:
            iq_list = areset.acquire_decimated(soc, readouts_per_experiment=1, load_pulses=True, progress=False)
            
        avgi, avgq = areset.acquire(soc, readouts_per_experiment=2, threshold=config["threshold"], load_pulses=True, progress=False)
        pre_reset.append(avgi[0][0]*config["readout_length"])
        post_reset.append(avgi[0][1]*config["readout_length"])
        
    fig, ax = plt.subplots()
    ax.plot(expt_pts,pre_reset,'s-', label="pre-reset")
    ax.plot(expt_pts,post_reset, 'o-', label="post-reset")
    ax.set_xlabel("Gain")
    ax.set_ylabel("Qubit Population")
    ax.set_title(title)
    
    if waveform==True:
        iq_list_grapher(iq_list)
                     
    return pre_reset, post_reset
        
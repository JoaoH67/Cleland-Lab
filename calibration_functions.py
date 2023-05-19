#Pyro 4 Preamble
#For now, this cell must be run in this notebook.
#Will be coming up with a fix later
import Pyro4
from qick import QickConfig
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION=4

ns_host = "192.168.0.174" #IP Address of QICK board
ns_port = 8888 #Change depending on which board you're connecting to
proxy_name = "myqick" #Change depending on how you initialized NS

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

#------------------------- Continuous Signal (TRY NOT TO USE IF POSSIBLE) -------------------------
class LoopbackProgram(AveragerProgram):
    def initialize(self):
        cfg=self.cfg   
        res_ch = cfg["qubit_ch"] #defines the output

        self.declare_gen(ch=cfg["qubit_ch"], nqz=cfg["nqz"]) #nqz sets the Nyquist zone. Preferrably, use nqz=1 for pulses under 3440MHz
        
        freq = self.freq2reg(cfg["pulse_freq"],gen_ch=res_ch) #converts frequency to generator frequency

        self.set_pulse_registers(ch=res_ch, style="const", length=cfg["length"], freq=freq, phase=0, gain=cfg["pulse_gain"], mode="periodic") #sets the pulse to be played
        #why don't we use self.set_iq?
        self.synci(200) #small delay to synchronize everything
    
    def body(self):
        
        self.pulse(ch=self.cfg["qubit_ch"]) #sends the pulse
        self.wait_all() #waits a specified number of clock ticks. Here, it's none.

def ContinuousSignal(ch=None, nqz=None, freq=None, gain=None, dbm=None):
    """Function to send a continuous signal of set frequency and gain
    
    :param ch: output channel [int]
    :param nqz: Nyquist zone [int]
    :param freq: frequency of the signal [MHz]
    :param gain: gain of the signal [a.u.]
    :param dbm: power of the signal [dBm]
    
    """ 

    if ch==None or nqz==None or freq==None:
        raise Exception("You must specify the channel, Nyquis zone, and frequency.")

    if (gain==None and dbm==None) and (gain is not None and dbm is not None):
        raise Exception("You must specify the signal power with EITHER gain or dbm.")

    if gain==None:
        gain=dbm2gain(dbm, freq, nqz, 3)
    config={"qubit_ch":ch,
            "reps":1,
            "relax_delay":1.0,
            "length":1000,
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

    prog =DCLoopbackProgram(soccfg, config)

    avgq, avgi = prog.acquire(soc, load_pulses=True, progress=True, debug=False)

class DCPulseLoopback(AveragerProgram):
    def initialize(self):
        cfg=self.cfg
        res_ch = cfg["dc_ch"] #defines the output
        
        self.add_pulse(ch=cfg["dc_ch"], name="dc", idata=gen_dc_waveform(cfg["dc_ch"], cfg["voltage"], shape="flat top", cycle_length=cfg["length"]))
        
        self.set_pulse_registers(ch=cfg["dc_ch"], outsel="input", style="arb", phase=0, 
                                freq=0, waveform="dc", gain=32767, mode="oneshot")

        self.synci(200) #small delay to synchronize everything
    
    def body(self):
        
        self.pulse(ch=self.cfg["dc_ch"]) #sends the pulse
        self.wait_all() #waits a specified number of clock ticks. Here, it's none.

def DCSignalPulse(ch=None, pulse_length=None, voltage=None):
    """Function to send a flat top pulse of variable length and voltage
    
    :param ch: output channel [int]
    :param pulse_length: length of flat top pulse [us]
    :param voltage: voltage [V]

    """ 

    if ch==None or voltage==None or pulse_length==None:
        raise Exception("You must specify the channel, pulse length, and voltage.")

    new_length = int(np.floor(soccfg.us2cycles(pulse_length,gen_ch=2)/16)*16)
    config={"dc_ch":ch,
            "reps":1,
            "relax_delay":1.0,
            "length":new_length,
            "voltage":voltage,
            "soft_avgs":1
           }

    prog =DCPulseLoopback(soccfg, config)

    avgq, avgi = prog.acquire(soc, load_pulses=True, progress=True, debug=False)

class RRProgram(RAveragerProgram):
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
    
        self.synci(200)
    
    def body(self):
        cfg=self.cfg
        
        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=4, mode="periodic")
        self.pulse(ch=cfg["dc_ch"])
        
        #Triggers waveform acquisition
        self.trigger(adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=cfg["adc_trig_offset"])
        
        self.pulse(ch=cfg["res_ch"])
        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))
        
    def update(self):
        self.mathi(self.r_rp, self.r_gain, self.r_gain, '+', self.cfg["step"])

def ROFrequency(q_name=None, ro_length=None, f_start=None, f_stop=None, f_expts=None, g_start=None, g_stop=None, g_expts=None, dc=0.0, plot=None, save=None, trial=None, collect=None, use_dbm=None):
    
    """
    finds the frequency of the readout resonator
    
    :param q_name: name of the qubit in dict_cfg
    :param ro_length: length of the probe pulse [us]
    :param f_start: start frequency [MHz]
    :param f_stop: stop frequency [MHz]
    :param f_expts: # of frequency points (int)
    :param g_start: start gain [a.u.] (min -32767)
    :param g_start: stop gain [a.u.] (max 32767)
    :param g_expts: number of gain steps
    :param dc: bias voltage [V]
    ...
    
    :params plot, save, trial: self-evident
    :param collect: if True, returns the data
    
    """

    if q_name==None or ro_length==None or f_start==None or f_stop==None or f_expts==None or g_start==None or g_stop==None or g_expts==None:
        raise Exception("You must specify q_name, ro_length, f_start, f_stop, f_expts, g_start, g_stop, and g_expts. To use dbm units for power, specify the gain measurements in units of dbm and set use_dbm=True")

    if use_dbm == True:
        g_start=int(dbm2gain(g_start, (f_start+f_stop)/2, 2, 3))
        g_stop=int(dbm2gain(g_stop, (f_start+f_stop)/2, 2, 3))

    expt_cfg={"reps":50, "relax_delay":10, "f_start":f_start, "f_stop":f_stop, "f_expts":f_expts, 
              "start":g_start, "step":int((g_stop-g_start)/g_expts), "expts":g_expts, "voltage":dc}

    dict_cfg = loadfrompickle(q_name)
    config={**dict_cfg, **expt_cfg}
    
    config["res_length"]=soccfg.us2cycles(ro_length,gen_ch=2) #converts length to clock cycles
    f_range=np.linspace(config["f_start"], config["f_stop"], config["f_expts"])

    amps=[]
    for i in tqdm(f_range):
        config["frequency"]=i,
        rspec=RRProgram(soccfg, config)
        expt_pts, avgi,avgq=rspec.acquire(soc, load_pulses=True, progress=False) #calls the previous cell with the actual pulse sequences and measurements
        amp=np.abs(avgi[0][0]+1j*avgq[0][0])
        amps.append(amp)
    expt_pts=np.array(expt_pts)
    amps=np.array(amps)
    background=np.mean(amps[0:int(f_expts/6)]) #we are interested in the relative amplitude, so we plot the data divided by the background
    
    if use_dbm==True:
        yvar=gain2dbm(expt_pts, (f_start+f_stop)/2, 2, 3)
        yvarname="Intensity (dBm)"
    else:
        yvar=expt_pts
        yvarname="DAC Gain (a.u.)"

    plotsave2d(plot=plot, save=save, q_name=q_name, title="RR Spectroscopy", trial=trial, xvar=f_range,
              xvarname="Frequency (MHz)", yvar=yvar, yvarname=yvarname, zdata=(amps/background).T, config=config, 
              zdataname="Transmission amplitude")
    
    if collect:
        return amps.T
    
    soc.reset_gens()

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
        
        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, 
                                freq=0, gain=cfg["start"], length=4, mode="periodic")
            
        self.set_pulse_registers(ch=cfg["res_ch"], style="flat_top", waveform="measure", freq=freq, length=cfg["res_length"], 
                                 phase=0, gain=cfg["pulse_gain"])
        
        self.synci(200)
    
    def body(self):
        cfg=self.cfg
        
        self.pulse(ch=cfg["dc_ch"])
            
        #Triggers waveform acquisition
        self.trigger(adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=cfg["adc_trig_offset"])
        
        self.pulse(ch=cfg["res_ch"])
        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))
        
    def update(self):
        self.mathi(self.r_rp, self.r_gain, self.r_gain, '+', self.cfg["step"])

class DCRRProgram1D(AveragerProgram):
    def initialize(self):
        cfg=self.cfg 
        self.declare_gen(ch=cfg["res_ch"], nqz=2)
        
        #Declare readout channels
        for ch in [0]:
            self.declare_readout(ch=ch, length=cfg["readout_length"],
                                 freq=cfg["frequency"], gen_ch=cfg["res_ch"])
            
        freq = self.freq2reg(cfg["frequency"], gen_ch=cfg["res_ch"], ro_ch=0)
        
        #Creates the waveform for the readout pulse
        self.add_gauss(ch=cfg["res_ch"], name="measure", sigma=cfg["res_length"]/8, length=cfg["res_length"]/2)
            
        self.set_pulse_registers(ch=cfg["res_ch"], style="flat_top", waveform="measure", freq=freq, length=cfg["res_length"], 
                                 phase=0, gain=cfg["pulse_gain"])
        
        self.synci(200)
    
    def body(self):
        cfg=self.cfg

        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=4, mode="periodic")
        self.pulse(ch=cfg["dc_ch"])
            
        #Triggers waveform acquisition
        self.trigger(adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=cfg["adc_trig_offset"], t=0)
        
        self.pulse(ch=cfg["res_ch"], t=0)
        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))

def DCROFrequency(q_name=None, ro_length=None, f_start=None, f_stop=None, f_expts=None, dc=None, dc_start=None, dc_stop=None, dc_expts=None, gain=None, use_dbm=None, plot=None, save=None, trial=None, collect=None):
    
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
    ...
    
    :params plot, save, trial: see plotsave2d function above
    :param collect: if True, returns the data
    
    """

    if q_name==None or ro_length==None or f_start==None or f_stop==None or f_expts==None or gain==None:
        raise Exception("You must specify q_name, ro_length, f_start, f_stop, f_expts and gain. To use dbm units for power, specify the gain measurements in units of dbm and set use_dbm=True")

    if use_dbm==True:
        gain=int(dbm2gain(gain, (f_start+f_stop)/2, 2, 3))
    
    expt_cfg={"reps":50, "relax_delay":10, "f_start":f_start, "f_stop":f_stop, "f_expts":f_expts, "pulse_gain":gain}
    
    dict_cfg = loadfrompickle(q_name)
    config={**dict_cfg, **expt_cfg}
    
    if dc==None:
        g_start=int(dc2gain(config["dc_ch"],dc_start))
        g_stop=int(dc2gain(config["dc_ch"], dc_stop))
        config["start"]=int(g_start)
        config["step"]=int((g_stop-g_start)/dc_expts)
        config["expts"]=dc_expts
        v_range=np.linspace(dc_start, dc_stop, dc_expts)
    else:
        config["voltage"]=dc
    
    
    config["res_length"]=soccfg.us2cycles(ro_length,gen_ch=2) #converts length to clock cycles
    f_range=np.linspace(config["f_start"], config["f_stop"], config["f_expts"])

    if dc==None:
        amps=[]
        for i in tqdm(f_range):
            config["frequency"]=i
            rspec=DCRRProgram(soccfg, config)
            expt_pts,avgi,avgq=rspec.acquire(soc, load_pulses=True, progress=False) #calls the previous cell with the actual pulse sequences and measurements
            amps.append(np.abs(avgi[0][0]+1j*avgq[0][0]))
        
        amps=np.array(amps)
        background=np.mean(amps[0:int(f_expts/6)]) #we are interested in the relative amplitude, so we plot the data divided by the background
        
        plotsave2d(plot=plot, save=save, q_name=q_name, title="RR-Spectroscopy-DC-Sweep", trial=trial, yvar=f_range,
                yvarname="Frequency (MHz)", xvar=v_range, xvarname="DC Bias (V)", zdata=(amps/background),
                zdataname="Transmission amplitude", config=config)

        if trial == None:
            trial = "01"
        datatracker = 'measurement_data/{}/Data-Stark-{}-{}-{}.csv'.format(q_name, "RR-Spectroscopy-DC-Sweep", date.today(), trial)
        
        soc.reset_gens()

        if collect:
            return amps.T, datatracker
        else:
            r_freq=[]
            for i in amps.T:
                f = np.argmin(i)
                r_freq.append(f_range[f])
            return np.vstack((v_range, r_freq)).T
    
    else:
        amps=[]
        for i in tqdm(f_range):
            config["frequency"]=i
            rspec=DCRRProgram1D(soccfg, config)
            avgi,avgq=rspec.acquire(soc, load_pulses=True, progress=False) #calls the previous cell with the actual pulse sequences and measurements
            amps.append(np.abs(avgi[0][0]+1j*avgq[0][0]))
        
        amps=np.array(amps)
        
        plotsave1d(plot=plot, save=save, q_name=q_name, title="RR-Spectroscopy-DC-Sweep-1D", trial=trial, xvar=f_range,
                xvarname="Frequency (MHz)", ydata=amps, ydataname="Transmission amplitude", config=config)
        
        if trial == None:
            trial = "01"
        datatracker = 'measurement_data/{}/Data-Stark-{}-{}-{}.csv'.format(q_name, "RR-Spectroscopy-DC-Sweep-1D", date.today(), trial)
        
        if collect:
            return amps, datatracker
    
    soc.reset_gens()

class PulseProbeSpectroscopyProgram(NDAveragerProgram):
    def initialize(self):
        cfg=self.cfg

        self.declare_gen(ch=cfg["qubit_ch"], nqz=2)
        self.declare_gen(ch=cfg["res_ch"], nqz=2)
        
        for ch in [0]:
            self.declare_readout(ch=ch, length=cfg["readout_length"],
                                 freq=cfg["f_res"], gen_ch=cfg["res_ch"])
             
        global freq
        freq = self.freq2reg(cfg["f_res"], gen_ch=cfg["res_ch"], ro_ch=0)
        
        global phase
        phase=self.deg2reg(cfg["phase"])
        
        self.add_DRAG(ch=cfg["qubit_ch"], name="drive", length=cfg["probe_length"], sigma=cfg["probe_length"]/4,
                      delta=cfg["delta"], alpha=cfg["alpha"])
        self.add_gauss(ch=cfg["res_ch"], name="measure", sigma=cfg["res_sigma"]/4, length=cfg["res_sigma"])
        
        self.res_r_freq = self.get_gen_reg(cfg["qubit_ch"], "freq")
        self.res_r_freq_update = self.new_gen_reg(cfg["qubit_ch"], init_val=cfg["f_start"], name="freq_update") 
        self.add_sweep(QickSweep(self, self.res_r_freq_update, cfg["f_start"], cfg["f_stop"], cfg["f_expts"]))

        self.sync_all(200)
    
    def body(self):
        cfg=self.cfg

        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=4, mode="periodic")
        self.pulse(ch=cfg["dc_ch"])
           
        #drive pulse
        self.set_pulse_registers(ch=cfg["qubit_ch"], style="arb", freq=0, 
                                 gain=cfg["qubit_gain"], waveform="drive", phase=0)
        self.res_r_freq.set_to(self.res_r_freq_update)
        self.pulse(ch=self.cfg["qubit_ch"])
        
        self.sync_all(cfg["drive_ro_delay"])

        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=4, mode="periodic")
        self.pulse(ch=cfg["dc_ch"])
        #measure pulse
        self.set_pulse_registers(ch=cfg["res_ch"], style="flat_top", waveform="measure", freq=freq, length=cfg["res_length"], 
                                 gain=cfg["res_gain"], phase=phase)
        self.trigger(adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=cfg["adc_trig_offset"])
        
        
        self.pulse(ch=cfg["res_ch"])
        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))

def QubitSpectroscopy(q_name=None, expt_type=None, probe_length=None, gain=None, use_dbm=None, reps=None, f_start=None, f_stop=None, f_expts=None, dc=None, f_res=None, plot=None, save=None, trial=None, collect=None):
    
    """
    finds the drive frequency of the qubit
    
    :param q_name: name of the qubit in dict_cfg
    :param exp_type: "Amplitude" or "Phase"
    :param probe_length: length of the probe pulse [us]
    :param gain: gain of the probe pulse [a.u. or dbm (see use_dbm)]
    :param use_dbm: use dbm units for gain inputs.
    :param reps: reps to be averaged over (int)
    :param f_start: start frequency [MHz]
    :param f_stop: stop frequency [MHz]
    :param f_expts: # of frequency points (int)
    :param dc: bias voltage [V]
    :param f_res: resonator frequency [MHz], if None: calculates f_res from most recent DCROFrequency1D experiment.
    ...
    
    :params plot, save, trial: see plotsave1d function above
    :param collect: if True, returns the data
    
    """

    if q_name==None or expt_type==None or probe_length==None or gain==None or reps==None or f_start==None or f_stop==None or f_expts==None or dc==None:
        raise Exception("You must specify q_name, ro_length, f_start, f_stop, f_expts, dc, and gain. To use dbm units for power, specify the gain measurements in units of dbm and set use_dbm=True")

    if use_dbm:
        gain=int(dbm2gain(gain, (f_start+f_stop)/2, 2, 3))


    expt_cfg={"reps": reps,"rounds":1,
              "probe_length":soccfg.us2cycles(probe_length, gen_ch=2), "qubit_gain":gain, "voltage":dc,
              "f_start":soccfg.freq2reg(f_start), "f_stop":soccfg.freq2reg(f_stop), "f_expts":f_expts
             }

    dict_cfg = loadfrompickle(q_name)
    config={**dict_cfg, **expt_cfg}
    if config["threshold"] is not None:
        corr=config["readout_length"]
    else:
        corr=1
    if f_res != None:
        config["f_res"]=f_res
    else:
        f_res = find_nearest(config["dc_arr"], dc)
        config["f_res"]=f_res-1
            
    amps=[]
    qspec=PulseProbeSpectroscopyProgram(soccfg, config)
    expt_pts, avgi, avgq = qspec.acquire(soc, threshold=config["threshold"], angle=None, load_pulses=True, progress=False, debug=False)
    amps.append((avgi[0][0]+1j*avgq[0][0])*corr)
    
    if expt_type=="Amplitude":
        ydata=np.abs(amps[0])
        ydataname="IQ amplitude"
    elif expt_type=="Phase":
        ydata=np.angle(amps[0])
        ydataname="IQ phase"

    plotsave1d(plot=plot, save=save,  q_name=q_name, title="Qubit Spectroscopy "+expt_type, trial=trial, xvar=np.linspace(f_start,f_stop,f_expts),
              xvarname="Qubit Frequency", ydata=ydata, ydataname=ydataname, config=config, data2save=amps)
              
    if collect:
        return amps
    
    soc.reset_gens()

class PulseProbeSpectroscopyZPAProgram(NDAveragerProgram):
    def initialize(self):
        cfg=self.cfg

        self.declare_gen(ch=cfg["qubit_ch"], nqz=2)
        self.declare_gen(ch=cfg["res_ch"], nqz=2)
        self.declare_gen(ch=cfg["dc_ch"], nqz=1)
        
        for ch in [0]:
            self.declare_readout(ch=ch, length=cfg["readout_length"],
                                 freq=cfg["f_res"], gen_ch=cfg["res_ch"])
             
        global freq
        freq = self.freq2reg(cfg["f_res"], gen_ch=cfg["res_ch"], ro_ch=0)
        
        global phase
        phase=self.deg2reg(cfg["phase"])
        
        gain = cfg["g0"]
        
        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", freq=0, phase=0, gain=gain, 
                                 length=4, mode="periodic")
        
        
        self.res_r_gain = self.get_gen_reg(cfg["dc_ch"], "gain")
        self.res_r_gain_update = self.new_gen_reg(cfg["dc_ch"], init_val=cfg["g_start"], name="gain_update") 
        
        self.add_sweep(QickSweep(self, self.res_r_gain_update, cfg["g_start"], cfg["g_stop"], cfg["g_expts"]))
        
        
        
        self.add_DRAG(ch=cfg["qubit_ch"], name="drive", length=cfg["probe_length"], sigma=cfg["probe_length"]/4,
                      delta=cfg["delta"], alpha=cfg["alpha"])
        self.add_gauss(ch=cfg["res_ch"], name="measure", sigma=cfg["res_sigma"]/4, length=cfg["res_sigma"])

        self.sync_all(200)
    
    def body(self):
        cfg=self.cfg

        self.res_r_gain.set_to(self.res_r_gain_update)
        self.pulse(ch=cfg["dc_ch"])
           
        #drive pulse
        self.set_pulse_registers(ch=cfg["qubit_ch"], style="arb", freq=self.freq2reg(cfg["qubit_freq"], gen_ch=cfg["qubit_ch"]), 
                                 gain=cfg["qubit_gain"], waveform="drive", phase=0)
        self.pulse(ch=self.cfg["qubit_ch"])
        
        self.sync_all(cfg["drive_ro_delay"])

        self.res_r_gain.set_to(self.cfg["g0"])
        self.pulse(ch=cfg["dc_ch"])
        #measure pulse
        self.set_pulse_registers(ch=cfg["res_ch"], style="flat_top", waveform="measure", freq=freq, length=cfg["res_length"], 
                                 gain=cfg["res_gain"], phase=phase)
        self.trigger(adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=cfg["adc_trig_offset"])
        
        
        self.pulse(ch=cfg["res_ch"])
        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))

def QubitSpectroscopyZPA(q_name=None, expt_type=None, probe_length=None, gain=None, use_dbm=None, reps=None, f_start=None, 
                         f_stop=None, f_expts=None, dc=None, zpa_start=None, zpa_stop=None, zpa_expts=None,
                         f_res=None, plot=None, save=None, trial=None, collect=None):
    
    """
    finds the drive frequency of the qubit
    
    :param q_name: name of the qubit in dict_cfg
    :param exp_type: "Amplitude" or "Phase"
    :param probe_length: length of the probe pulse [us]
    :param gain: gain of the probe pulse [a.u. or dbm (see use_dbm)]
    :param use_dbm: use dbm units for gain inputs.
    :param reps: reps to be averaged over (int)
    :param f_start: start frequency [MHz]
    :param f_stop: stop frequency [MHz]
    :param f_expts: # of frequency points (int)
    :param dc: bias voltage [V]
    :param f_res: resonator frequency [MHz], if None: calculates f_res from most recent DCROFrequency1D experiment.
    ...
    
    :params plot, save, trial: see plotsave1d function above
    :param collect: if True, returns the data
    
    """

    if q_name==None or expt_type==None or probe_length==None or gain==None or reps==None or f_start==None or f_stop==None or f_expts==None or dc==None:
        raise Exception("You must specify q_name, ro_length, f_start, f_stop, f_expts, dc, and gain. To use dbm units for power, specify the gain measurements in units of dbm and set use_dbm=True")

    if use_dbm:
        gain=int(dbm2gain(gain, (f_start+f_stop)/2, 2, 3))


    expt_cfg={"reps": reps,"rounds":1,
              "probe_length":soccfg.us2cycles(probe_length, gen_ch=2), "qubit_gain":gain, "voltage":dc,
              
             }

    dict_cfg = loadfrompickle(q_name)
    config={**dict_cfg, **expt_cfg}
    if config["threshold"] is not None:
        corr=config["readout_length"]
    else:
        corr=1
    if f_res != None:
        config["f_res"]=f_res
    else:
        f_res = find_nearest(config["dc_arr"], dc)
        config["f_res"]=f_res-1
    
    config["g0"]=int(dc2gain(config["dc_ch"], config["voltage"]))
    config["g_start"]=int(dc2gain(config["dc_ch"], config["voltage"]+zpa_start))
    config["g_stop"]=int(dc2gain(config["dc_ch"], config["voltage"]+zpa_stop))
    config["g_expts"]=zpa_expts
    amps=[]
    for i in tqdm(np.linspace(f_start,f_stop,f_expts)):
        config["qubit_freq"]=i
        qspec=PulseProbeSpectroscopyZPAProgram(soccfg, config)
        expt_pts, avgi, avgq = qspec.acquire(soc, threshold=config["threshold"], angle=None, load_pulses=True, progress=False, debug=False)
        amps.append((avgi[0][0]+1j*avgq[0][0])*corr)
    amps=np.array(amps)
    
    if expt_type=="Amplitude":
        zdata=np.abs(amps)
        zdataname="IQ amplitude"
    elif expt_type=="Phase":
        zdata=np.angle(amps)
        zdataname="IQ phase"
        
    yvar=gain2dc(config["dc_ch"], expt_pts[0])
    
    plotsave2d(plot=plot, save=save,  q_name=q_name, title="ZPA Spectroscopy "+expt_type, trial=trial, xvar=np.linspace(f_start,f_stop,f_expts),
              xvarname="Qubit Frequency", yvar=yvar, yvarname="Voltage", zdata=zdata.T, zdataname=zdataname, config=config)
              
    if collect:
        return amps
    
    soc.reset_gens()

class RabiProgram(AveragerProgram):
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
        self.add_gauss(ch=cfg["res_ch"], name="measure", sigma=cfg["res_sigma"]/4, length=cfg["res_sigma"])
            
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
        
        self.sync_all(cfg["drive_ro_delay"])
        
        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, 
                                 gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])), length=4, mode="periodic")
        
        self.pulse(ch=cfg["dc_ch"])
        
        self.set_pulse_registers(ch=cfg["res_ch"], style="flat_top", waveform="measure", freq=f_res, length=cfg["res_length"], 
                                 gain=cfg["res_gain"], phase=phase)
        
        self.trigger(adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=cfg["adc_trig_offset"])
        
        self.pulse(ch=cfg["res_ch"])
        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))

class NDRabiProgram(NDAveragerProgram):
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
        self.add_gauss(ch=cfg["res_ch"], name="measure", sigma=cfg["res_sigma"]/4, length=cfg["res_sigma"])
        
        self.res_r_gain = self.get_gen_reg(cfg["qubit_ch"], "gain")
        self.res_r_gain_update = self.new_gen_reg(cfg["qubit_ch"], init_val=cfg["g_start"], name="gain_update") 
        self.add_sweep(QickSweep(self, self.res_r_gain_update, cfg["g_start"], cfg["g_stop"], cfg["g_expts"]))
        
        
        self.synci(200)
        
    def body(self):
        cfg=self.cfg
        
        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=4, mode="periodic")
        self.pulse(ch=cfg["dc_ch"])

        f_res=self.freq2reg(cfg["f_res"], gen_ch=cfg["qubit_ch"], ro_ch=0)
        f_ge=self.freq2reg(cfg["f_ge"], gen_ch=cfg["qubit_ch"])
        
        self.set_pulse_registers(ch=cfg["qubit_ch"], style="arb", waveform="drive", freq=f_ge, gain=0, phase=0)
        self.res_r_gain.set_to(self.res_r_gain_update)
        self.pulse(ch=cfg["qubit_ch"])
        
        self.sync_all(cfg["drive_ro_delay"])
        
        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, 
                                 gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])), length=4, mode="periodic")
        
        self.pulse(ch=cfg["dc_ch"])
        
        self.set_pulse_registers(ch=cfg["res_ch"], style="flat_top", waveform="measure", freq=f_res, length=cfg["res_length"], 
                                 gain=cfg["res_gain"], phase=phase)
        
        self.trigger(adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=cfg["adc_trig_offset"])
        
        self.pulse(ch=cfg["res_ch"])
        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))

def Rabi(q_name=None, qubit_gain=None, qubit_length=None, l_start=None, l_stop=None, l_expts=None,
         g_start=None, g_stop=None, g_expts=None, 
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
        raise Exception("You must specify gain (Length Rabi) or length (Amplitude Rabi), not both.")
    
    if qubit_gain==None and qubit_length==None:
        fit=None
    
    dict_cfg = loadfrompickle(q_name)
    
    if qubit_gain!=None:
        expt_cfg={
            "qubit_gain":qubit_gain,
            "l_start":soccfg.us2cycles(l_start), "l_stop":soccfg.us2cycles(l_stop), "l_expts":l_expts, "reps":reps, "voltage":dc
           }
        config={**dict_cfg, **expt_cfg}
        xvar=np.linspace(l_start, l_stop, l_expts)*1000+9.3
        xvarname="Pulse length (ns)"

    elif qubit_length!=None:
        expt_cfg={
            "qubit_length":soccfg.us2cycles(qubit_length),
            "g_start":g_start, "g_stop":g_stop, "g_expts":g_expts, "reps":reps, "voltage":dc
           }
        config={**dict_cfg, **expt_cfg}
        xvar=np.linspace(g_start, g_stop, g_expts)
        xvarname="Gain"
    
    else:
        expt_cfg={
            "l_start":soccfg.us2cycles(l_start), "l_stop":soccfg.us2cycles(l_stop), "l_expts":l_expts,
            "g_start":g_start, "g_stop":g_stop, "g_expts":g_expts, "reps":reps, "voltage":dc
           }
        config={**dict_cfg, **expt_cfg}
        xvar=np.linspace(g_start, g_stop, g_expts)
        xvarname="Gain"
        yvar=np.linspace(l_start, l_stop, l_expts)
        yvarname="Pulse length (ns)"

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
    
    elif qubit_length!=None:
        for l in tqdm(np.linspace(config["g_start"], config["g_stop"], config["g_expts"])):
            config["qubit_gain"]=int(l)
            rabi=RabiProgram(soccfg, config)
            avgi,avgq = rabi.acquire(soc, threshold=config["threshold"], load_pulses=True, progress=False,debug=False)
            results.append(np.sqrt(avgi[0][0]**2+avgq[0][0]**2)*corr)
    else:
        for l in tqdm(np.linspace(config["l_start"], config["l_stop"], config["l_expts"])+4):
            config["qubit_length"]=l
            rabi=NDRabiProgram(soccfg, config)
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
        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=4, mode="periodic")

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
        return avgi0, avgq0, avgi1, avgq1
    
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
        
        for ch in [0]:
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

def T1(q_name=None, start=None, stop=None, expts=None, reps=None, dc=None, plot=None, save=None, trial=None, collect=None, fit=None):
    
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
    if q_name==None or start==None or stop==None or expts==None or reps==None or dc==None:
        raise Exception("You must specify q_name, start, stop, expts, reps, and dc")

    step=(stop-start)/expts
    expt_cfg={ "start":start, "step":step, "expts":expts, "reps":reps, "voltage":dc,
            "relax_delay":250
           }
    dict_cfg = loadfrompickle(q_name)
    config={**dict_cfg, **expt_cfg}

    if config["threshold"] is not None:
        corr=config["readout_length"]
    else:
        corr=1
    t1p=T1Program(soccfg, config)

    x_pts, avgi, avgq = t1p.acquire(soc, threshold=config["threshold"], load_pulses=True, progress=True, debug=False)
    
    results=np.sqrt(avgi[0][0]**2+avgq[0][0]**2)*corr

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
        
        self.synci(200)
        
    def body(self):
        cfg=self.cfg
        
        w_iter = 0
        for i in self.cfg["gate_sequence"]:    
            if i == 9: #Pause in pulses
                self.sync_all(cfg["wait_sequence"][w_iter])
                w_iter += 1
            elif i == 0: #Identity gate
                self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=cfg["pulse_length"], mode="periodic")
                self.pulse(ch=cfg["dc_ch"])
                self.set_pulse_registers(ch=self.cfg["qubit_ch"], freq=f_ge, gain=0, style="arb",waveform="drive", phase=phase)
                self.pulse(ch=self.cfg["qubit_ch"])
                self.sync_all(cfg["sync_time"])
            elif i == 1: #X
                self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=cfg["pulse_length"], mode="periodic")
                self.pulse(ch=cfg["dc_ch"])
                self.set_pulse_registers(ch=self.cfg["qubit_ch"], freq=f_ge, gain=self.cfg["pi_gain"], style="arb", 
                                         waveform="drive", phase=phase)
                self.pulse(ch=self.cfg["qubit_ch"])
                self.sync_all(cfg["sync_time"])
            elif i == 2: #Y
                self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=cfg["pulse_length"], mode="periodic")
                self.pulse(ch=cfg["dc_ch"])
                self.set_pulse_registers(ch=self.cfg["qubit_ch"], freq=f_ge, phase=yphase, gain=self.cfg["pi_gain"], 
                                         style="arb", waveform="drive")
                self.pulse(ch=self.cfg["qubit_ch"])
                self.sync_all(cfg["sync_time"])
            elif i == 3: #X2
                self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=cfg["pulse_length"], mode="periodic")
                self.pulse(ch=cfg["dc_ch"])
                self.set_pulse_registers(ch=self.cfg["qubit_ch"], freq=f_ge, gain=self.cfg["pi2_gain"], 
                                         style="arb", waveform="drive", phase=phase)
                self.pulse(ch=self.cfg["qubit_ch"])
                self.sync_all(cfg["sync_time"])
            elif i == 4: #Y2
                self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=cfg["pulse_length"], mode="periodic")
                self.pulse(ch=cfg["dc_ch"])
                self.set_pulse_registers(ch=self.cfg["qubit_ch"], freq=f_ge, phase=yphase,
                                     gain=self.cfg["pi2_gain"], style="arb", waveform="drive")
                self.pulse(ch=self.cfg["qubit_ch"])
                self.sync_all(cfg["sync_time"])
            elif i == 5: #-X2
                self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=cfg["pulse_length"], mode="periodic")
                self.pulse(ch=cfg["dc_ch"])
                self.set_pulse_registers(ch=self.cfg["qubit_ch"], freq=f_ge, phase=negphase, gain=self.cfg["pi2_gain"],
                                         style="arb", waveform="drive")
                self.pulse(ch=self.cfg["qubit_ch"])
                self.sync_all(cfg["sync_time"])
            elif i == 6: #-Y2
                self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=cfg["pulse_length"], mode="periodic")
                self.pulse(ch=cfg["dc_ch"])
                self.set_pulse_registers(ch=self.cfg["qubit_ch"], freq=f_ge, phase=negyphase,
                                     gain=self.cfg["pi2_gain"], style="arb", waveform="drive")
                self.pulse(ch=self.cfg["qubit_ch"])
                self.sync_all(cfg["sync_time"])
            elif i == 7: #-X
                self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=cfg["pulse_length"], mode="periodic")
                self.pulse(ch=cfg["dc_ch"])
                self.set_pulse_registers(ch=self.cfg["qubit_ch"], freq=f_ge, gain=self.cfg["pi_gain"], phase=negphase, 
                                         style="arb", waveform="drive")
                self.pulse(ch=self.cfg["qubit_ch"])
                self.sync_all(cfg["sync_time"])
            elif i == 8: #-Y
                self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=cfg["pulse_length"], mode="periodic")
                self.pulse(ch=cfg["dc_ch"])
                self.set_pulse_registers(ch=self.cfg["qubit_ch"], freq=f_ge, phase=negyphase, 
                                         gain=self.cfg["pi_gain"], style="arb", waveform="drive")
                self.pulse(ch=self.cfg["qubit_ch"])
                self.sync_all(cfg["sync_time"])
            else:
                raise ValueError("'cfg[gate_sequence] value does not correspond to basic gate generator [0,8] or a pause 9 '")
        #Readout
        self.sync_all(cfg["drive_ro_delay"])

        self.set_pulse_registers(ch=cfg["dc_ch"], style="const", phase=0, freq=0, gain=int(dc2gain(cfg["dc_ch"], cfg["voltage"])),
                                 length=cfg["res_length"]+cfg["res_sigma"], mode="periodic")
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

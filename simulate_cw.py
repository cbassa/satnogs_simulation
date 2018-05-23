#!/usr/bin/env python
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# Morse alphabet
morse = {'a':'.-', 'b':'-...', 'c':'-.-.', 'd':'-..', 'e':'.', 'f':'..-.', 
         'g':'--.', 'h':'....', 'i':'..', 'j':'.---', 'k':'-.-', 'l':'.-..', 
         'm':'--', 'n':'-.', 'o':'---', 'p':'.--.', 'q':'--.-', 'r':'.-.', 
         's':'...', 't':'-', 'u':'..-', 'v':'...-', 'w':'.--', 'x':'-..-', 
         'y':'-.--', 'z':'--..', '1':'.----', '2':'..---', '3':'...--', '4':'....-', 
         '5':'.....', '6':'-....', '7':'--...', '8':'---..', '9':'----.', '0':'-----'}

def encode_string(string):
    morse_code=[]
    for char in string:
        if char==' ':
            morse_code.append('/')
        else:
            morse_code.append(morse[char])

    return ' '.join(morse_code)

def heaviside_step(x, x0, k):
    return 0.5+0.5*np.tanh(k*(x-x0))

def dit(x, x0, k, w):
    return heaviside_step(x, x0, k)-heaviside_step(x, x0+w, k)

def generate_waveform(x, string, x0, k, w):
    y=np.zeros_like(x)
    for char in string:
        if char=='.':
            y+=dit(x, x0, k, w)
            x0+=2*w
        elif char=='-':
            y+=dit(x, x0, k, 3*w)
            x0+=4*w
        elif char==' ':
            x0+=3*w
        elif char=='/':
            x0+=3*w
            
    return y

def read_satellite_track(fname):
    d = np.loadtxt(fname)
    t = d[:, 0]
    r = d[:, 1]
    v = d[:, 2]
    return t, r, v

if __name__ == "__main__":
    # Settings
    samp_rate = 20000     # Sample rate in Hz
    morse_samp_rate = 200 # Morse code sample rate in Hz
    dit_width = 0.06      # Dit width in seconds
    sat_freq = 145.9e6    # Satellite transmission frequency in Hz
    ramp = 1000.0         # Dit ramp (edge fall off)
    tmax = 837.0          # Signal length in seconds
    amp_noise = 1.0       # Noise amplitude
    amp_morse = 1.0      # Morse amplitude at 1000 km range

    # String to decode
    string = 20*"the quick brown fox jumps over the lazy dog 0123456789        "
    code = encode_string(string)

    # Generate coarsely sampled Morse code waveform
    td = np.linspace(0, tmax, tmax*morse_samp_rate)
    wd = generate_waveform(td, code, 1.0, ramp, dit_width)
    #wd = np.ones_like(td)

    # Read satellite track
    sat_t, sat_r, sat_v = read_satellite_track("satellite_track.dat")

    # Set up interpolation functions
    fr = interpolate.interp1d(sat_t, sat_r)
    fv = interpolate.interp1d(sat_t, sat_v)
    fw = interpolate.interp1d(td, wd)
    
    # Generate signal
    t = np.linspace(0, tmax, tmax*samp_rate)

    # Compute range and velocity
    r = fr(t)
    v = fv(t)
    freq = -sat_freq*v/299792.458
    amp = amp_morse*(1000.0/r)**2
    phase = 2.0*np.pi*np.cumsum(freq/samp_rate)
    
    # Generate waveform
    w = amp*fw(t)

    plt.figure(figsize=(20,5))
    plt.plot(t,phase)
    plt.show()

    
    # Generate noise
    y = np.random.normal(0.0, amp_noise, 2*len(t))
    
    # Add waveform
    y += np.vstack((w*np.cos(phase), w*np.sin(phase))).reshape((-1),order="F")

    # Write binary file
    y.astype('float32').tofile("waveform.dat")


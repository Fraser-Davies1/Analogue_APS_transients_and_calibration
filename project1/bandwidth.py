from pytestlab.instruments import AutoInstrument
import os
import time
import numpy as np
from pathlib import Path

from pytestlab.bench import Bench
from pytestlab.measurements.session import MeasurementSession
import polars as pl



from pytestlab.instruments.Multimeter import DMMFunction  # or wherever DMMFunction lives in your install


# loading the instruments
osc = AutoInstrument.from_config("keysight/DSOX1204G")
awg = AutoInstrument.from_config("keysight/EDU33212A")
psu = AutoInstrument.from_config("keysight/EDU36311A")
dmm = AutoInstrument.from_config("keysight/EDU34450A")


# printing IDN of instruments
print(osc.id())
print(awg.id())
print(psu.id())
print(dmm.id())




measurements = []


channel=1


f = np.logspace(3, np.log10(20e6), num=44)
f = np.unique(np.rint(f).astype(int))


awg.set_function(channel, function_type="SIN")
awg.set_output_load_impedance(channel, "INF")
awg.set_amplitude(channel, amplitude=0.1)
awg.set_offset(channel, offset=4)
awg.set_output_state(channel, "ON")


osc.set_channel_axis(1, 20e-3, 0.585)
osc.set_acquisition_time(1e-3)


for freq in f:

        
    awg.set_frequency(2, frequency=freq)
    print(f"set freq {freq}")
    time.sleep(1)


    samples = []
    for _ in range(10):
        vpp_out = osc.measure_voltage_peak_to_peak(1)
        samples.append(vpp_out)
        time.sleep(0.1)

    vpp_mean = float(np.mean(samples))


    measurements.append({
        'set_freq': freq,
        'Vpp_out': vpp_mean
    })

    print(f"vpp {vpp_mean}")

df = pl.DataFrame(measurements)

# Save to CSV
df.write_csv("bandwidth.csv")

print("Saved to bandwidth.csv")
print(df)
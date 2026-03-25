from pytestlab import Bench
import time

with Bench.open("bench.yaml") as bench:
    # Set maximum light
    bench.psu.channel(2).set(voltage=5.0).on()
    time.sleep(2)
    
    # Query current reading on scope
    v_drop = bench.osc._query(":MEASure:VAVerage? CHANnel2")
    print(f"DEBUG: Current V_drop at 5V input: {v_drop}V")
    
    # Check if the channel is actually on
    ch_state = bench.osc._query(":CHANnel2:DISPlay?")
    print(f"DEBUG: Scope CH2 Display State: {ch_state}")

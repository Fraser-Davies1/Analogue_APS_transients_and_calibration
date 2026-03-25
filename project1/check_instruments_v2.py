from pytestlab import Bench

with Bench.open("bench.yaml") as bench:
    print(f"OSC: {bench.osc.id()}")
    try:
        from pytestlab import AutoInstrument
        psu = AutoInstrument.from_config("keysight/EDU36311A")
        psu.connect_backend()
        print(f"PSU: {psu.id()}")
        psu.close()
    except Exception as e:
        print(f"PSU Error: {e}")

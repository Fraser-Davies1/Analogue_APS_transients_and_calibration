from pytestlab import Bench

with Bench.open("bench.yaml") as bench:
    print(f"OSC: {bench.osc.id()}")
    try:
        print(f"WFG: {bench.wfg.id()}")
    except Exception as e:
        print(f"WFG Error: {e}")

from pytestlab import Bench

with Bench.open("bench.yaml") as bench:
    ch = bench.wfg.channel(1)
    print(f"Facade methods: {[m for m in dir(ch) if not m.startswith('_')]}")

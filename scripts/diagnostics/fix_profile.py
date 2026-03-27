import yaml
from pathlib import Path

# Path to original profile
orig_path = Path('/home/coder/project/.venv/lib/python3.14/site-packages/pytestlab/profiles/keysight/EDU33212A.yaml')
with open(orig_path, 'r') as f:
    data = yaml.safe_load(f)

# Patch the schema mismatch
for ch in data['channels']:
    if 'frequency' in ch:
        ch['frequency_range'] = ch.pop('frequency')
    if 'amplitude' in ch:
        ch['amplitude_range'] = ch.pop('amplitude')
    if 'dc_offset' in ch:
        ch['offset_range'] = ch.pop('dc_offset')

# Save as a local fixed profile
with open('EDU33212A_fixed.yaml', 'w') as f:
    yaml.dump(data, f)

print("Created EDU33212A_fixed.yaml")

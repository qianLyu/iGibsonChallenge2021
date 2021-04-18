import torch
from collections import OrderedDict
import json
import os

PTH_PATH = ''

print('Loading model...')
state_dict = torch.load(
    PTH_PATH,
    map_location='cuda'
)
print('Done loading model.')
print('Converting to JSON...')
actual_dict = OrderedDict()
for k, v in state_dict["state_dict"].items():
	actual_dict[k] = v.tolist()
output_name = os.path.basename(PTH_PATH)[:-3]+'json'
with open(output_name, 'w') as f:
	json.dump(actual_dict, f)
print('JSON weights saved to '+output_name)
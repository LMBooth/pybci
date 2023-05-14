import re
import yaml
import ast
from os import path

# Read the setup.py file
with open('setup.py', 'r') as f:
    setup_content = f.read()

# Extract the metadata using regular expressions
metadata = re.findall(r"^\s*([\w]+)\s*=\s*(.+)$", setup_content, re.MULTILINE)
metadata_dict = {}

# Evaluate the values and assign them to the metadata dictionary
for key, value in metadata:
    try:
        eval_value = ast.literal_eval(value)
        metadata_dict[key] = eval_value
    except (ValueError, SyntaxError):
        metadata_dict[key] = value.strip("'\"")

# Write the metadata to a YAML file
with open('setup.yml', 'w') as f:
    yaml.dump(metadata_dict, f, default_flow_style=False)

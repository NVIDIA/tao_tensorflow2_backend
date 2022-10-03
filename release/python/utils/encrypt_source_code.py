# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
"""Helper utils for using pyarmor to encrypt .py files"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os

def flatten_dictionary_keys(json_dict, model_to_entry_files_map, prefix="", depth=0):
    """Flatten recursive dictionary keys into a single key separated by /"""
    for dict_key,value in sorted(json_dict.items(),key=lambda x: x[0]):
        if isinstance(value, dict):
            flatten_dictionary_keys(value, model_to_entry_files_map, os.path.join(prefix,dict_key),depth+1)
        else:
            model_to_entry_files_map[os.path.join(prefix,dict_key)] = ','.join(value)

def pyarmor_init(model_to_entry_files_map, root):
    """Assign entry files and generate pyarmor config files for all modules"""
    for model_path, entry_scripts in model_to_entry_files_map.items():
        os.system("pyarmor init --src " + os.path.join(root, model_path) + " --entry '" + entry_scripts + "' " + os.path.join("/obf_src", model_path)  )
        os.chdir(os.path.join("/obf_src", model_path))
        obf_code_flag = ""
        if "pointpillars" in model_path:
            obf_code_flag = "--obf-code=0"
        call_command = f"pyarmor config --manifest 'global-include *.py' --is-package 0 --enable-suffix 1 {obf_code_flag}"
        os.system(call_command)

def generate_pyarmor_runtime_file(model_to_entry_files_map):
    """Generate pyarmor shared object file"""
    os.chdir("/")
    for model_path, entry_scripts in model_to_entry_files_map.items():
        os.system("pyarmor build --output /dist --only-runtime " + os.path.join("/obf_src", model_path))
        break

def generate_encrypted_files(model_to_entry_files_map):
    """Generate encrypted files for all modules"""
    os.chdir("/")
    for model_path, entry_scripts in model_to_entry_files_map.items():
        os.system("pyarmor build --output " + os.path.join("/dist",model_path)+" --no-runtime "+ os.path.join("/obf_src",model_path))

def encrypt_files(root):
    # Get the absolute path of json file
    pyarmor_entry_files_json = os.path.dirname(os.path.abspath(__file__))+"/pyarmor_entry_files.json"

    # Load the json file
    json_obj = json.load(open(pyarmor_entry_files_json,"r"))

    # Convert the dictionary in a processable format
    model_to_entry_files_map = {}
    flatten_dictionary_keys(json_obj, model_to_entry_files_map)

    # Initiate pyarmor for all the available modules
    pyarmor_init(model_to_entry_files_map, root)

    # Generate a runtime pyrarmor.so file 
    generate_pyarmor_runtime_file(model_to_entry_files_map)

    # Write encrypted files onto disk  
    generate_encrypted_files(model_to_entry_files_map)
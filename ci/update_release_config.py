import sys
import yaml

def update_yaml(yaml_file, key_value):
    try:
        with open(yaml_file, 'r') as file:
            yaml_data = yaml.safe_load(file)

        key, value = key_value.split(':', 1)
        key = key.strip()
        value = value.strip()

        yaml_data[key] = value

        with open(yaml_file, 'w') as file:
            yaml.dump(yaml_data, file, default_flow_style=False)

        print(f"Updated {key} in {yaml_file} to '{value}'")
    except Exception as e:
        print(f"Error updating YAML file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <yaml_file> <key_value>")
        sys.exit(1)

    yaml_file = sys.argv[1]
    key_value = sys.argv[2]

    update_yaml(yaml_file, key_value)

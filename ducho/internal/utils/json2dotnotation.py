# Json leaves to be formatted as list in output
leaves = ("reshape", "output_layers")

# This function can be used to parse a .json file
# def parse_json_to_dot_notation(json_file_path):
#   # with open(json_file_path, 'r') as file:
#   #   data = json.load(file)
#   parse_and_print(json_file_path)

output = ""
def parse_and_print(obj, prefix="", output=output):
  if isinstance(obj, dict):
    for key, value in obj.items():
      new_prefix = f"{prefix}{key}."
      output = parse_and_print(value, new_prefix, output)
  elif isinstance(obj, list):
    if prefix[:-1].endswith(leaves):
        formatted_value = format_leaf_value(obj)
        output += f"{prefix[:-1]}: {formatted_value}\n"
        #print(f"{prefix[:-1]}: {formatted_value}")
    else:
      for i, item in enumerate(obj):
        new_prefix = f"{prefix}{i}."
        output = parse_and_print(item, new_prefix, output)
  else:
    formatted_value = format_leaf_value(obj)
    output += f"{prefix[:-1]}: {formatted_value}\n"
    #print(f"{prefix[:-1]}: {formatted_value}")
  return output

def format_leaf_value(value):
  if isinstance(value, list):
    return "[" + ", ".join(map(str, value)) + "]"
  elif isinstance(value, dict):
    return ""  # To prevent printing intermediate nodes
  else:
    return value


# Ducho Banner
banner = '''
      ██████╗  ██╗   ██╗  ██████╗ ██╗  ██╗  ██████╗ 
      ██╔══██╗ ██║   ██║ ██╔════╝ ██║  ██║ ██╔═══██╗
      ██║  ██║ ██║   ██║ ██║      ███████║ ██║   ██║
      ██║  ██║ ██║   ██║ ██║      ██╔══██║ ██║   ██║
      ██████╔╝ ╚██████╔╝ ╚██████╗ ██║  ██║ ╚██████╔╝
      ╚═════╝   ╚═════╝   ╚═════╝ ╚═╝  ╚═╝  ╚═════╝ 
'''
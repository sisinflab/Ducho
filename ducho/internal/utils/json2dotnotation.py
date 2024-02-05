# Json leaves to be formatted as list in output
leaves = ("reshape", "output_layers", "mean", "std")

output = ""
def parse_and_print(obj, prefix="", output=output):
  """
  Recursively parse a JSON object and convert it to dot notation string.

  Args:
      obj: The JSON object to parse.
      prefix (str, optional): The prefix for dot notation representation. Defaults to "".
      output (str, optional): The output string. Defaults to the outer scope variable 'output'.

  Returns:
      str: The dot notation string representation.
  """
  if isinstance(obj, dict):
    for key, value in obj.items():
      new_prefix = f"{prefix}{key}."
      output = parse_and_print(value, new_prefix, output)
  elif isinstance(obj, list):
    if prefix[:-1].endswith(leaves):
        formatted_value = format_leaf_value(obj)
        output += f"{prefix[:-1]}: {formatted_value}\n"
    else:
      for i, item in enumerate(obj):
        new_prefix = f"{prefix}{i}."
        output = parse_and_print(item, new_prefix, output)
  else:
    formatted_value = format_leaf_value(obj)
    output += f"{prefix[:-1]}: {formatted_value}\n"
  return output

def format_leaf_value(value):
  """
  Format a leaf value for dot notation string representation.

  Args:
      value: The value to format.

  Returns:
      str: The formatted leaf value.
  """
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
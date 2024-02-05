import re

# This code is inspired by the human sorting algorithm described at
# https://nedbatchelder.com/blog/200712/human_sorting.html

def _if_int(text):
    return int(text) if text.isdigit() else text


def _natural_keys(text):
    return [_if_int(c) for c in re.split(r'(\d+)', text)]


def human_sort(unsorted_list):
    """
    Sorts a list of strings both alphabetically and numerically.

    This function sorts the list in such a way that it follows both the alphabetical order and the numerical order.
    For example, ['10','2','1','8'] will be sorted as ['1','2','8','10'].

    Args:
        unsorted_list (List[str]): The list of strings to sort.

    Returns:
        List[str]: The sorted list of strings.
    """
    unsorted_list.sort(key=_natural_keys)
    return unsorted_list

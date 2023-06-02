import re

# this code is inspired by
# https://nedbatchelder.com/blog/200712/human_sorting.html


def _if_int(text):
    return int(text) if text.isdigit() else text


def _natural_keys(text):
    return [_if_int(c) for c in re.split(r'(\d+)', text)]


def human_sort(unsorted_list):
    """
    It sorts a list of string both alphabetically and numerically. This means that the order follow the alphabet order
    but also the cardinal one. E.g: ['10','2','1','8'] -> ['1','2','8','10']
    :param unsorted_list: the list of string to sort
    :return: the sorted list of string
    """
    unsorted_list.sort(key=_natural_keys)
    return unsorted_list

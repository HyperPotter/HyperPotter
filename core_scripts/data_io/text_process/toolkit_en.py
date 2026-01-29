
import os
import sys
import re

from core_scripts.data_io.text_process import toolkit_all
_pad = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_skip_symbols = ['_', '~']
_arpabet_symbols_raw = [
    'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 
    'AH', 'AH0', 'AH1', 'AH2', 'AO', 'AO0', 'AO1', 'AO2', 
    'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
    'B',  'CH',  'D',   'DH',  'EH', 'EH0', 'EH1', 'EH2', 
    'ER', 'ER0', 'ER1', 'ER2', 'EY', 'EY0', 'EY1', 'EY2', 
    'F',  'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 
    'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 
    'OW1', 'OW2', 'OY', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 
    'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 
    'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
_arpabet_symbol_marker = '@'
_arpabet_symbols = [_arpabet_symbol_marker + x for x in _arpabet_symbols_raw]
_symbols = [_pad] + list(_special) + list(_punctuation) \
           + list(_letters) + _arpabet_symbols
_symbol_to_index = {y: x+1 for x, y in enumerate(_symbols)}

def symbol_num():
    return len(_symbols)

def symbol2index(x):
    return _symbol_to_index[x]

def index2symbol(x):
    return _symbols[x-1]
_whitespace_re = re.compile(r'\s+')

_number_map = {'1': 'one', '2': 'two', '3': 'three',
               '4': 'four', '5': 'five', '6': 'six', 
               '7': 'seven', '8': 'eight', '9': 'nine', '0': 'zero'}

def text_numbers(text):
    """ Place holder, just convert individual number to alphabet
    """
    def _tmp(tmp_text):
        if all([x in _number_map for x in tmp_text]):
            return ' '.join([_number_map[x] for x in tmp_text])
        else:
            return tmp_text
    tmp = ' '.join([_tmp(x) for x in text.split()])
    if text.startswith(' '):
        tmp = ' ' + tmp
    return tmp

def text_case_convert(text):
    """ By default, use lower case
    """
    return text.lower()

def text_whitespace_convert(text):
    return re.sub(_whitespace_re, ' ', text)

def text_normalizer(text):
    return text_whitespace_convert(text_numbers(text_case_convert(text)))


def flag_convert_symbol(symbol):
    """ check whether input symbol should be converted or not

    input
    -----
      symbol: str
    
    output
    ------
      bool
    """
    return symbol in _symbol_to_index and symbol not in _skip_symbols

def rawtext2indices(text):
    """ Look up the table and return the index for input symbol in input text
    
    input
    -----
      text: str
    
    output
    ------
      list of indices

    for example, 'text' -> [23, 16, 28, 23]
    """
    return [symbol2index(x) for x in text if flag_convert_symbol(x)]

def arpabet2indices(arpa_text):
    """ Look up the table and return the index for input symbol in input text

    input
    -----
      arpa_text: str
    
    output
    ------
      list of indices

    for example, 'AH HH' -> [12 19]
    """
    tmp = [_arpabet_symbol_marker + x for x in arpa_text.split()]
    return [symbol2index(x) for x in tmp if flag_convert_symbol(x)]


def text2code(text):
    """ Convert English text and ARPAbet into code symbols (int)
    """
    if text.startswith(toolkit_all._curly_symbol):
        return arpabet2indices(text.lstrip(toolkit_all._curly_symbol))
    else:
        text_normalized = text_normalizer(text)
        return rawtext2indices(text_normalized)

def code2text(codes):
    txt_tmp = [index2symbol(x) for x in codes]
    txt_tmp = ''.join(txt_tmp)
    return text_whitespace_convert(txt_tmp.replace(_arpabet_symbol_marker, ' '))
    


if __name__ == "__main__":
    print("Definition of text processing toolkit for English")

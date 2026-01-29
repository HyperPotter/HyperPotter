import os
import sys
import re

_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

_curly_symbol = '{'

def parse_curly_bracket(text):
    text_list = []
    text_tmp = text

    while len(text_tmp):
        re_matched = _curly_re.match(text_tmp)
        
        if re_matched:
            text_list.append(re_matched.group(1))
            text_list.append(_curly_symbol + re_matched.group(2))
            text_tmp = re_matched.group(3)
        else:
            text_list.append(text_tmp)
            break
    return text_list


if __name__ == "__main__":
    print("Definition of text processing tools for all languages")

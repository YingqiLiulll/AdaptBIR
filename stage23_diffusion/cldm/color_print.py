class Style:
    END = '\33[0m'
    BOLD = '\33[1m'
    ITALIC = '\33[3m'
    URL = '\33[4m'
    BLINK = '\33[5m'
    BLINK2 = '\33[6m'
    SELECTED = '\33[7m'

class Font:
    BLACK = '\33[30m'
    RED = '\33[31m'
    GREEN = '\33[32m'
    YELLOW = '\33[33m'
    BLUE = '\33[34m'
    VIOLET = '\33[35m'
    BEIGE = '\33[36m'
    WHITE = '\33[37m'

class Background:
    BLACK = '\33[40m'
    RED = '\33[41m'
    GREEN = '\33[42m'
    YELLOW = '\33[43m'
    BLUE = '\33[44m'
    VIOLET = '\33[45m'
    BEIGE = '\33[46m'
    WHITE = '\33[47m'

# https://www.jb51.net/article/258799.htm
def highlight(string, fcolor='', bgcolor='', style=''):
    fcolor_code = getattr(Font, fcolor.upper(), '')
    bgcolor_code = getattr(Background, bgcolor.upper(), '')
    style_code = getattr(Style, style.upper(), '')
    return f"{style_code}{fcolor_code}{bgcolor_code}{string}{Style.END}"

def cprint(s, **kwargs):
    args = ["fcolor", "bgcolor", "style"]
    for arg in args:
        if arg in kwargs:
            v = kwargs.pop(arg)
            s = highlight(s, **{arg: v})
    print(s, **kwargs)

def warn(s):
    cprint(s, fcolor="red", bgcolor="red", style="bold")

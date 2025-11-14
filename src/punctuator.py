# src/punctuator.py
def simple_punct(text):
    if not text:
        return ''
    s = text.strip()
    s = s[0].upper() + s[1:] if len(s) > 0 else s
    if s and s[-1] not in '.!?':
        s += '.'
    return s

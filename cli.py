import sys
import cfg

def parse_value(value):
    if value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    elif value == "True":
        return True
    elif value == "False":
        return False
    elif '.' in value:
        try: return float(value)
        except ValueError: pass
    try: return int(value)
    except ValueError: return value

def args():
    ans = {}
    for arg in sys.argv[1:]:
        eq = arg.find('=')
        if eq >= 0:
            name = arg[:eq].strip()
            value = parse_value(arg[eq+1:].strip())
        else:
            name = arg.strip()
            value = True
        ans[name] = value

    # Use 'cfg' only to determine config file; remove it from overrides
    cfgfile = ans.pop('cfg', 'private/cfg.json')
    try:
        config = cfg.cfg(file=cfgfile)
    except:
        config = {}

    # Command-line values override config file
    return {**config, **ans}
def safe_json(val):
    import json
    try:
        json.dumps(val)
        return val
    except TypeError:
        return str(val)

def process_dict(d):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            new_d[k] = process_dict(v)
        elif isinstance(v, list):
            new_d[k] = [safe_json(item) if not isinstance(item, (dict, list)) else (process_dict(item) if isinstance(item, dict) else item) for item in v]
        else:
            new_d[k] = safe_json(v)
    return new_d

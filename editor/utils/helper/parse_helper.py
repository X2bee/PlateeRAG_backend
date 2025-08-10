def parse_param_value(value):
    if not isinstance(value, str):
        return value

    if not value or not value.strip():
        return value

    value = value.strip()

    lower_val = value.lower()
    if lower_val in ('true', 't', 'yes'):
        return True
    elif lower_val in ('false', 'f', 'no'):
        return False

    if lower_val in ('none', 'null', 'nil'):
        return None

    try:
        if value.lstrip('+-').isdigit():
            return int(value)

        float_val = float(value)
        if float_val.is_integer() and '.' not in value and 'e' not in lower_val:
            return int(float_val)

        return float_val

    except ValueError:
        pass

    return value

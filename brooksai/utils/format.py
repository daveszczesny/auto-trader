def format_time(seconds: int) -> str:
    weeks = seconds // (7 * 24 * 3600)
    seconds %= (7 * 24 * 3600)
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    if weeks > 0:
        return f'{weeks} weeks, {days} days, {hours} hours, {minutes} minutes, {seconds} seconds'
    elif days > 0:
        return f'{days} days, {hours} hours, {minutes} minutes, {seconds} seconds'
    elif hours > 0:
        return f'{hours} hours, {minutes} minutes, {seconds} seconds'
    elif minutes > 0:
        return f'{minutes} minutes, {seconds} seconds'
    else:
        return f'{seconds} seconds'
from math import log


def backups_to_keep(n, a=2, b=2):
    if n == 0:
        return set()
    elif n == 1:
        return {1}
    else:
        return {n} | backups_to_keep(n - a ** (int(log(n, b)) - 1), a, b)

def keep_things_at_day(n, a=2, b=2):
    return {int(n-x) for x in backups_to_keep(n, a, b)}


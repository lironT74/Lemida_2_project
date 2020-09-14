
def create_month_dict(data):
    months = []
    for x in data:
        month = x[0, -2]
        if month not in months:
            months.append(month)
    month_dict = {val: i for i, val in enumerate(sorted(months))}
    return month_dict

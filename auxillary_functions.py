
def get_x_any_y(df, dates):
    x, y = [], []
    for date in dates:
        day_df = df[df['date'] == date]
        x.append(day_df.iloc[:, [2, 3, 4, 5, 6, 7, 8, 9, 11, 12]].to_numpy())
        y.append(day_df.iloc[:, 10].to_numpy())
    return x, y

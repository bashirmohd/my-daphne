save_csv_and_plot.py:
(1) assumes that all price files (names as given) are in a folder named “prices” reachable from current directory
(2) assumes that curtailment file (named hourly_curtailment.xlsx) is in current directory
(3) plot_curtail_and_price takes in as arguments: the site name (in the csv), its corresponding price column name (in the csv), and cur (the name of the df)
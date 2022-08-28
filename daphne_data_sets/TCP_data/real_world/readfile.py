
import os
import pandas as pd


data = pd.read_csv('fb.csv',nrows=1000)
data_list = data.values.tolist() #creates a list of the first 1000 rows (excludes header)
print(data_list)


file_to_convert=os.path.join("fb.csv")
shorter_file=os.path.join("fb1.csv")

with open(file_to_convert, 'r') as data:
  plaintext = data.read(nrows=1000)

plaintext = plaintext.replace('          ', ',')
plaintext = plaintext.replace('         ', ',')
plaintext = plaintext.replace('        ', ',')
plaintext = plaintext.replace('       ', ',')
plaintext = plaintext.replace('      ', ',')
plaintext = plaintext.replace('     ', ',')
plaintext = plaintext.replace('    ', ',')
plaintext = plaintext.replace('   ', ',')
plaintext = plaintext.replace('  ', ',')
plaintext = plaintext.replace(' ', ',')



with open(file_to_convert, 'w') as data:
  data.write(plaintext)





#run_converter.py
#this file runs converters.py located in \SURGE\surge\datagen\converters.py
#Good for converting data types, as well as in this case delimiters.

#################################
#Owen's example UCI Wine Dataset
#################################
from surge.datagen.converters import convert_csv_delimiter
convert_csv_delimiter(

    r"C:\Users\Bipo1\Downloads\UCI Wine Data\winequality-red.csv",
    r"C:\Users\Bipo1\Downloads\UCI Wine Data\winequality-red-comma.csv",
)

convert_csv_delimiter(
    r"C:\Users\Bipo1\Downloads\UCI Wine Data\winequality-white.csv",
    r"C:\Users\Bipo1\Downloads\UCI Wine Data\winequality-white-comma.csv",
)
#this script obtains the number of unique conditions/diseases in a file, mainly just for statistics purposes (e.g. to write in the final paper when describing CHARDAT's dataset statistics)

import sys
import json

in_file = sys.argv[1]

def get_diseases_count(in_file):
    f = open(in_file,'r',encoding='UTF-8')
    lines = [x.strip() for x in f.readlines()]
    unique_conditions = []
    for line in lines:
        condition = line.split(' <sep> ')[1]
        if condition not in unique_conditions:
            unique_conditions.append(condition)
    print(f"Number of unique conditions for {in_file}: {len(unique_conditions)}")

get_diseases_count(in_file)
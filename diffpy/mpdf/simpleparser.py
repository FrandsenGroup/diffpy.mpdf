#!/usr/bin/env python
##############################################################################
#
# diffpy.mpdf         by Frandsen Group
#                     Benjamin A. Frandsen benfrandsen@byu.edu
#                     (c) 2022 Benjamin Allen Frandsen
#                      All rights reserved
#
# File coded by:    Victor Velasco
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################

"""Parser that can be used for CIF and MCIF files."""

import re

class SimpleParser:
    """Extract information from a text file, typically CIF or MCIF.


    This class reads in a given text file (typically a CIF or MCIF file) and
    pulls out relevant information in the form of a python dictionary.
    """
    def __init__(self, file_dict):
        self.file_dict = file_dict

    def ReadFile(self, filename):
        file = open(filename, "r")

        while True:
            line = file.readline()

            if not line:
                break

            line = line.split()

            if not line:
                continue

            self.parse(line, file)

        file.close()
        return self.file_dict

    def parse(self, line, file):
        if line[0].find("_") == 0:
            self.insert_dictionary(line)

        elif line[0] == "loop_": 
            keys = []
            values = {} 
            counter = 0

            while True:
                l = file.readline()
                l = l.strip()
                if not l:
                    break
                if l.find("_") == 0:
                    keys.append(l)
                else:
                    if '[' in l:
                            l = l.split(' ',1)
                            l[1] = l[1].strip('[]')
                            temp = l[1].split()
                            l[1] = ''
                            for i, char in enumerate(temp):
                                l[1] = l[1] + char
                                if i < len(temp) - 1:
                                        l[1] = l[1] + ','
                    else:
                        l = l.split()
                    values[counter] = l 
                    counter += 1
            self.insert_loop(keys, values)

    def insert_loop(self, keys, values): 
        for i in values:
            for k in range(len(keys)):
                if keys[k] in self.file_dict:
                    self.file_dict[keys[k]].append(self.to_numeric(values[i][k]))
                    continue
        
                self.file_dict[keys[k]] = [self.to_numeric(values[i][k])]

    def insert_dictionary(self, line):
        if line[0] in self.file_dict:
            self.file_dict[line[0]].append(self.to_numeric(self.to_string(line)))
            return 
        
        self.file_dict[line[0]] = [self.to_numeric(self.to_string(line))]

    def to_string(self, line):
        str_1 = ""
        if len(line) > 1:
            if line[1][0] == '"':
                line[1] = line[1].strip('"')
                line[len(line)-1] = line[len(line)-1].strip('"')
            elif line[1][0] == "'":
                line[1] = line[1].strip("'")
                line[len(line)-1] = line[len(line)-1].strip("'")
        for i in range (1, len(line)):
            str_1 += line[i] + " "
                
        return str_1.strip()

    def to_numeric(self, x):
        if re.search(r'\(\d+\)$',x):
            x = x[:x.index('(')]
        if x.isdigit() == True or self.check_float(x) == True:
            return float(x) 
        return x

    def check_float(self, num):
        try:
            float(num)

            return True
        
        except ValueError:
            return False
    

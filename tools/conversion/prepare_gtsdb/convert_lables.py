import os

# Mapping dictionary from GTSDB
# old:new (matching train.yaml)
mapping = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 3,
    6: 3,
    7: 3,
    8: 3,
    9: 9,
    10: 3,
    11: 11,
    12: 3,
    13: 13,
    14: 14,
    15: 15,
    16: 16,
    17: 17,
    18: 18,
    19: 19,
    20: 20,
    21: 21,
    22: 22,
    23: 23,
    24: 24,
    25: 25,
    26: 26,
    27: 27,
    28: 28,
    29: 29,
    30: 30,
    31: 31,
    32: 32,
    33: 33,
    34: 34,
    35: 35,
    36: 36,
    37: 37,
    38: 38,
    39: 39,
    40: 40,
    41: 10,
    42: 12,
    43: 8,
    44: 6,
    45: 7,
    46: 4,
    47: 5,
    48: 500,
}

def modify_first_number(file_path, mapping):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            try:
                number = int(parts[0])
                if number in mapping:
                    parts[0] = str(mapping[number])
                modified_lines.append(' '.join(parts) + '\n')
            except ValueError:
                modified_lines.append(line)

    with open(file_path, 'w') as file:
        file.writelines(modified_lines)

# Specify the directory containing the .txt files
directory = r'C:\Users\Benedikt Seeger\Downloads\gtsdb'

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        modify_first_number(file_path, mapping)

print("Modification complete.")

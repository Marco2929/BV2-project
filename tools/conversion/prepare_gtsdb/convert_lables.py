import os

# Mapping dictionary from GTSDB
# old:new (matching train.yaml)
mapping = {
    0: 3,
    1: 4,
    2: 5,
    3: 6,
    4: 7,
    5: 8,
    6: 47,
    7: 10,
    8: 12,
    9: 13,
    10: 14,
    11: 15,
    12: 16,
    13: 17,
    14: 11,
    15: 18,
    16: 19,
    17: 20,
    18: 21,
    19: 22,
    20: 23,
    21: 24,
    22: 25,
    23: 26,
    24: 27,
    25: 28,
    26: 29,
    27: 30,
    28: 31,
    29: 32,
    30: 33,
    31: 34,
    32: 35,
    33: 36,
    34: 37,
    35: 38,
    36: 39,
    37: 40,
    38: 41,
    39: 42,
    40: 43,
    41: 44,
    42: 45,
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
directory = r'C:\Users\Benedikt Seeger\Downloads\gtsdb_new'

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        modify_first_number(file_path, mapping)

print("Modification complete.")

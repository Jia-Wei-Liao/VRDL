
def read_classes(path):
    with open(path, 'r') as f:
        classes_list = [word.strip() for word in f.readlines()]

    return classes_list

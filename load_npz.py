import numpy as np
from numpy import load
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', default='/data/cqu5/CLIP-Driven-Universal-Model/out/dataset_02/test_healthp_0/predict/02_TCIA_Pancreas-CTlabel0002.npz', help='The path to npz file')
    args = parser.parse_args()

    data = load(args.file_name)
    lst = data.files
    for item in lst:
        print(item)
        
        print(data[item].shape)

if __name__ == "__main__":
    main()
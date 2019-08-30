import pandas as pd
import numpy as np

from optparse import OptionParser
# from reproduceability import reproduceability


def get_args():
    parser = OptionParser()
    parser.add_option('--ensemble', dest='ensemble', default=False, help='file you want to ensemble')
    parser.add_option('--threshold', type="float", dest='threshold', default=-1, help='threshold')
    parser.add_option('--tofile', dest='tofile', default='submission.csv', help='file location you get')

    (options, args) = parser.parse_args()
    return options


def load_args():
    args = get_args()
    if not args.ensemble:
        raise("Please specify files you want to ensemble")
    else:
        files = args.ensemble.replace(", ", ",").split(",")
    return args.threshold, files, args.tofile

if __name__ == '__main__':
    """
    PLAYGROUND
    """
    threshold, files, tofile = load_args()
    index_id = "Id"
    attribute_id = "attribute_ids"
    # reproduceability()

    files = [pd.read_csv(file, delimiter=',', encoding="utf-8-sig", engine='python').set_index(index_id).sample(frac=1).sort_values(index_id, axis=0, ascending=True) for file in files]
    index = files[0].index
    keys = files[0].keys()
    files = np.stack([file.values for file in files], axis=0) # (n, 7443, 1103)
    files = np.average(files, axis=0, weights=None)

    if threshold < 0 or threshold > 1:
        print("Threshold {} was set improperly. So I will not pass threshold for you.")
        df = pd.DataFrame(files, index=index, columns=keys)
        print(df.head())
    else:
        files = (files>threshold).astype(np.byte)
        prediction = []
        for i in range(files.shape[0]):
            args = np.argwhere(files[i] == 1).reshape(-1).tolist()
            prediction.append(" ".join(list(map(str, args))))
        df = pd.DataFrame({attribute_id:prediction}, index=index)
        print(df.head())

    df.to_csv(tofile, encoding='utf-8', index=True)

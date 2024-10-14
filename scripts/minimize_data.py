"""
Load data and export proprioceptive data only.
"""

import pickle

import debug


def minimze_data(data):
    # keep same structure
    new_data = {}
    for key, value in data.items():
        if "usb_cam" in key:
            continue
        assert isinstance(value, list)
        assert value[0].ndim == 1
        new_data[key] = value
    return new_data


@debug.iex
def minimize_file(file):
    print(f"Load {file}...")
    with open(file, "rb") as f:
        data = pickle.load(f)
    new_data = minimze_data(data)
    new_file = file + ".min.pkl"
    print(f"Save {new_file}...")
    with open(new_file, "wb") as f:
        pickle.dump(new_data, f)
    print("Done")


def main():
    file = "data/pkl/test/2024-07-26-21-15-44.pkl"
    minimize_file(file)
    import pdb; pdb.set_trace()


assert __name__ == "__main__"
main()

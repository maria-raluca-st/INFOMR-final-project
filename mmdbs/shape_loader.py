import vedo

from itertools import cycle, islice
from pathlib import Path

ORIGINAL_SHAPEDIR = Path("./shapes")
CLASSES = [_dir.stem for _dir in ORIGINAL_SHAPEDIR.glob("*")]


def main():
    cls = choose_class()
    objs = (ORIGINAL_SHAPEDIR / cls).iterdir()
    # Currently only showing the first one
    vedo.load(str(next(objs))).show().close()


def choose_class() -> str:
    batch_size = 4
    print(f"n: next {batch_size}.")
    for classes in batched(cycle(CLASSES), batch_size):
        print(*enumerate(classes))
        i = input()
        if i == 'n':
            continue
        return classes[int(i)]
    

def batched(iterable, n):
    # Taken from Python 3.12
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch

if __name__ == '__main__':
    main()

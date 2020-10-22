import argparse


def part_1():
    pass


def part_2():
    pass


def part_3():
    pass


def part_4():
    pass


def part_5():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part_1', action='store_true')
    parser.add_argument('--part_2', action='store_true')
    parser.add_argument('--part_3', action='store_true')
    parser.add_argument('--part_4', action='store_true')
    parser.add_argument('--part_5', action='store_true')

    if parser.part_1:
        part_1()

    if parser.part_2:
        part_2()

    if parser.part_3:
        part_3()

    if parser.part_4:
        part_4()

    if parser.part_5:
        part_5()

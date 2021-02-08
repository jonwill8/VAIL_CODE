"""
This fun script is used to print the collatz length of all numbers up to a user defined bound
"""

import numpy as np
import matplotlib.pyplot as plt


def calc_collatz_len(n):
    """
    This function is used to calculate the collatz length of any number n
    """
    collatz_len = 0
    if n == 1:  # if n = 1, collatz length is 1
        return collatz_len
    else:
        while n != 1:  # looping until n is equal to 1
            if n % 2 == 0:  # n is even
                n = n / 2
                collatz_len += 1
            elif n % 2 == 1:  # if n is odd
                n = 3 * n + 1
                collatz_len += 1
    return collatz_len


def calc_collatz_paths(numlist):
    """
    This function is used to calculate the collatz path of each number in the passed numlist
    """
    num_collatz_lens = [0] * len(numlist)  # creating zero array the length of the passed numlist

    for index, num in enumerate(numlist):
        num_collatz_lens[index] = calc_collatz_len(
            num)  # assigning each index in num_collatz_lens arr to proper collatz path

    return num_collatz_lens


if __name__ == '__main__':
    # prompting user for upper bound
    user_bound = int(input('Enter the upper number bound: '))
    num_list = np.linspace(1, user_bound, user_bound)
    collatz_lens = calc_collatz_paths(num_list)
    # plotting the collatz path
    plt.plot(num_list, collatz_lens)
    # labeling axes
    plt.xlabel('Integers')
    plt.ylabel('Respective Collatz Paths')
    # creating tile
    plt.title(f'Graph of Collatz paths of all integers up until {user_bound}')
    # showing our plot
    plt.show()
    # printing num w/ the highest collatz path to terminal
    print(f'Until the bound {user_bound}, the number {int(num_list[collatz_lens.index(max(collatz_lens))])} '
          f'has the longest collatz path of {max(collatz_lens)}')
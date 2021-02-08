"""
This script can be used to approximate the value of pi
"""

# importing numpy & matplotlib & random modules
import random
import math
import numpy as np
import matplotlib.pyplot as plt


def point_in_circle(x, y):
    """
    This function returns a bool that determines if the passed point
    is within the area bound of a circle defined by the EQ:
    (x-0.5)^2 + (y-0.5)^2 = (0.5)^2
    """
    return ((x - 0.5)**2 + (y - 0.5)**2) <= (0.5**2)


def gen_random_points():
    """
    This function returns a random x/y point from the bounds 0 --> 1
    """
    return [random.random(), random.random()]


def simulate_random_points(num_points):
    """
    This function creates an array of length num_points of randomly generated x/y points
    """
    return [gen_random_points() for _ in range(num_points)]


def calc_point_proportion(point_arr):
    """
    This function determines the proportion of points from point_arr that are w/i the bounds of the circle
    """
    return len([[x, y] for [x, y] in point_arr if point_in_circle(x, y)]) / len(point_arr)


def approximate_pi(num_points):
    """
    This function is used to return an approximation of pi based on a simulation of n = num_points
    """
    # pulling our array of randomly generated points
    rand_points = simulate_random_points(num_points)
    # calculating the proportion of points within of circle bounds
    point_portion = calc_point_proportion(rand_points)
    # returning our pi approximation.
    # Note: Formula --> pi = (point proportion * area of square)/(enclosed circle radius^2).
    # Note: area of square = 1. Circle Radius = 1/2
    pi_approx = (point_portion * 1) / (0.5 ** 2)
    return pi_approx


if __name__ == '__main__':
    num_points_bound = int(input('Enter the number of points you wish to use to approximate pi: '))
    # pulling an array of pi approximations using varying # of points
    pi_approximations = [approximate_pi(num_points) for num_points in range(1, num_points_bound + 1)]
    # creating our x values of the # of points used to approximate pi
    points = np.arange(1, num_points_bound + 1, 1)
    # plotting the real value of pi
    plt.plot(points, [math.pi] * num_points_bound)
    # plotting our approximations of pi
    plt.plot(points, pi_approximations)
    # labeling plot accordingly
    plt.title(f'Approximation of Pi Using # of Points Ranging from 1 to {num_points_bound}')
    plt.xlabel('Number of Random Points Used in Approximation')
    plt.ylabel('Respective Pi Approximations')
    plt.show()
    # printing our overall average pi approximation + residual from real value
    average_pi_approx = np.average(pi_approximations)
    print(f'Our overall average pi approximation was: {average_pi_approx}')
    print(f'Residual of our overall average pi approximation is: {math.pi - average_pi_approx}')
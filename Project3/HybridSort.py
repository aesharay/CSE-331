"""
Name: Aesha Ray
Project 3 - Hybrid Sorting
Developed by Sean Nguyen and Andrew Haas
Based on work by Zosha Korzecke and Olivia Mikola
CSE 331 Spring 2021
Professor Sebnem Onsay
"""
from typing import TypeVar, List, Callable
import math

T = TypeVar("T")            # represents generic type


def merge_sort(data: List[T], threshold: int = 0,
               comparator: Callable[[T, T], bool] = lambda x, y: x <= y) -> int:
    """
    Given a list of values, perform a merge sort to sort the list and calculate the inversion count.
    When a threshold is provided, use a merge sort algorithm until the partitioned lists are smaller
    than or equal to the threshold
    Then use insertion sort.
    """
    if len(data) < 2:
        return 0
    if len(data) < threshold:
        insertion_sort(data, comparator)

    mid = len(data) // 2
    start = data[:mid]
    end = data[mid:]
    inv = merge_sort(start, threshold, comparator) + merge_sort(end, threshold, comparator)

    for i in range(0, len(data)):
        j = 0
        while i + j < len(data):
            if j == len(end):
                data[i + j] = start[i]
                i += 1
            elif i < len(start) and comparator(start[i], end[j]):
                data[i + j] = start[i]
                i += 1
            else:
                data[i + j] = end[j]
                j += 1
                inv += len(start) - i
        if threshold != 0:
            return 0
        return inv

def insertion_sort(data: List[T],
                   comparator: Callable[[T, T], bool] = lambda x, y: x <= y) -> None:
    """
    Given a list of values and comparator, count the number of insertions on the list
    """
    for i in range(1, len(data)):
        j = i
        while j > 0 and comparator(data[j], data[j - 1]):
            temp = data[j]
            data[j] = data[j - 1]
            data[j - 1] = temp
            j -= 1

def hybrid_sort(data: List[T], threshold: int,
                comparator: Callable[[T, T], bool] = lambda x, y: x <= y) -> None:
    """
    Wrapper function to call merge_sort() as a Hybrid Sorting Algorithm.
    Call merge_sort() with provided threshold, and comparator function.
    """
    merge_sort(data, threshold, comparator)

def inversions_count(data: List[T]) -> int:
    """
    Wrapper function to call merge_sort() on a copy of data to retrieve the inversion count.
    Should call merge_sort() with no threshold, and the default comparator.
    """
    result = 0
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            if data[i] > data[j]:
                result += 1
    return result

def reverse_sort(data: List[T], threshold: int) -> None:
    """
    Wrapper function to use merge_sort() to sort the data in reverse.
    Should call merge_sort() with provided threshold, and a comparator you define.
    """
    comparator: Callable[[T, T], bool] = lambda x, y: x > y
    merge_sort(data, threshold, comparator)

def password_rate(password: str) -> float:
    """
    Rate a given password via the equation given in the problem statement
    """
    chars = len(password)
    unique = []
    for i in password[::]:
        if i not in unique:
            unique.append(i)
    unique_num = len(unique)
    inversions = inversions_count(list(password))
    return math.sqrt(chars) * math.sqrt(unique_num) + inversions

def password_sort(data: List[str]) -> None:
    """
    Sort a list of passwords by their ratings (the results from password_rate())
    """
    tups = []
    for i in range(0, len(data)):
        tups.append((data[i], password_rate(data[i])))
    hybrid_sort(tups, 0, lambda x, y: x[1] > y[1])

    for i in range(0, len(tups)):
        data[i] = tups[i][0]

"""
Project 4
CSE 331 S21 (Onsay)
Name
CircularDeque.py
"""

from __future__ import annotations
from typing import TypeVar, List

# for more information on typehinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")                                # represents generic type
CircularDeque = TypeVar("CircularDeque")        # represents a CircularDeque object


class CircularDeque:
    """
    Class representation of a Circular Deque
    """

    __slots__ = ['capacity', 'size', 'queue', 'front', 'back']

    def __init__(self, data: List[T] = [], capacity: int = 4):
        """
        Initializes an instance of a CircularDeque
        :param data: starting data to add to the deque, for testing purposes
        :param capacity: amount of space in the deque
        """
        self.capacity: int = capacity
        self.size: int = len(data)

        self.queue: list[T] = [None] * capacity
        self.front: int = None
        self.back: int = None

        for index, value in enumerate(data):
            self.queue[index] = value
            self.front = 0
            self.back = index

    def __str__(self) -> str:
        """
        Provides a string represenation of a CircularDeque
        :return: the instance as a string
        """
        if self.size == 0:
            return "CircularDeque <empty>"

        string = f"CircularDeque <{self.queue[self.front]}"
        current_index = self.front + 1 % self.capacity
        while current_index <= self.back:
            string += f", {self.queue[current_index]}"
            current_index = (current_index + 1) % self.capacity
        return string + ">"

    def __repr__(self) -> str:
        """
        Provides a string represenation of a CircularDeque
        :return: the instance as a string
        """
        return str(self)

    # ============ Modify below ============ #

    def __len__(self) -> int:
        """
        Returns the size of a deque
        """
        return self.size

    def is_empty(self) -> bool:
        """
        Checks if deque is empty
        If empty, return true, else return false
        """
        return len(self) == 0

    def front_element(self) -> T:
        """
        Returns the first element in the deque
        If deque is empty, return None
        """
        if self.is_empty():
            return None
        return self.queue[self.front]

    def back_element(self) -> T:
        """
        Returns the last element in the deque
        If deque is empty, return None
        """
        if self.is_empty():
            return None
        return self.queue[self.back]

    def front_enqueue(self, value: T) -> None:
        """
        Inserts an element in the front of a deque
        """
        if self.is_empty():
            self.front, self.back = 0, 0
            self.queue[self.front] = value
        else:
            self.front = (self.front - 1) % self.capacity
            self.queue[self.front] = value
        self.size += 1
        if self.size == self.capacity:
            self.grow()

    def back_enqueue(self, value: T) -> None:
        """
        Inserts an element in the back of a deque
        """
        if self.is_empty():
            self.front, self.back = 0, 0
            self.queue[self.back] = value
        else:
            self.back = (self.back + 1) % self.capacity
            self.queue[self.back] = value
        self.size += 1
        if self.size == self.capacity:
            self.grow()

    def front_dequeue(self) -> T:
        """
        Removes and returns the first element in the deque
        Return None, if deque is empty
        """
        if self.is_empty():
            return None
        first = self.queue[self.front]
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        if (self.size <= (self.capacity // 4)) and ((self.capacity // 2) >= 4):
            self.shrink()
        return first

    def back_dequeue(self) -> T:
        """
        Removes and returns the last element in the deque
        Return None if deque is empty
        """
        if self.is_empty():
            return None
        first = self.queue[self.back]
        self.back = (self.back - 1) % self.capacity
        self.size -= 1
        if (self.size <= (self.capacity // 4)) and ((self.capacity // 2) >= 4):
            self.shrink()
        return first

    def grow(self) -> None:
        """
        If the current size is equal to the current capacity,
        double the capacity of the circular deque
        """
        if self.size == self.capacity:
            new_queue = self.queue
            self.queue = []
            for i in range(self.front, self.size + self.front):
                new_front = i % self.capacity
                self.queue.append(new_queue[new_front])
            self.capacity *= 2
            self.queue.extend([None] * (self.capacity - self.size))
            self.front = 0
            self.back = self.size - 1

    def shrink(self) -> None:
        """
        If the current size is less than or equal to one fourth the current capacity,
        and 1/2 the current capacity is greater than or equal to 4,
        halves the capacity.
        """
        if (self.size <= (self.capacity // 4)) and ((self.capacity // 2) >= 4):
            new_queue = self.queue
            self.queue = []
            for i in range(self.front, self.size + self.front):
                new_front = i % self.capacity
                self.queue.append(new_queue[new_front])
            self.capacity //= 2
            self.queue.extend([None] * (self.capacity - self.size))
            self.front = 0
            self.back = self.size - 1

def LetsPassTrains102(infix: str) -> str:
    """
    regex = r"\-*\d+\.\d+|\-\d+|[\(\)\-\^\*\+\/]|(?<!-)\d+|\w"
    ops = {'*': 3, '/': 3,  # key: operator, value: precedence
           '+': 2, '-': 2,
           '^': 4,
           '(': 0}  # '(' is lowest bc must be closed by ')'
    """
    result = ""
    self = CircularDeque()
    for i in infix:
        if i.isalpha(i):
            self.back_enqueue(self, i)
        elif i == '(':
            self.back_enqueue(i)
        elif i == ')':
            while not self.is_empty() and self.front != '(':
                a = self.back_dequeue()
                self.back_enqueue(self, a)
            if not self.is_empty() and self.front != '(':
                return str(-1)
            else:
                self.back_dequeue()
    while not self.is_empty():
        self.back_enqueue(self, self.back_dequeue())
    print(self)

"""
Aesha Ray
Project 5 - PriorityHeaps - Solution Code
CSE 331 Fall 2020
Dr. Sebnem Onsay
"""

from typing import List, Any
#from PriorityNode import PriorityNode, MaxNode, MinNode
from Project7.PriorityNode import PriorityNode, MaxNode, MinNode


class PriorityQueue:
    """
    Implementation of a priority queue - the highest/lowest priority elements
    are at the front (root). Can act as a min or max-heap.
    """

    #   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   DO NOT MODIFY the following attributes/functions
    #   Modify only below indicated line
    __slots__ = ["_data", "_is_min"]

    def __init__(self, is_min: bool = True):
        """
        Constructs the priority queue
        :param is_min: If the priority queue acts as a priority min or max-heap.
        """
        self._data = []
        self._is_min = is_min

    def __str__(self) -> str:
        """
        Represents the priority queue as a string
        :return: string representation of the heap
        """
        return F"PriorityQueue [{', '.join(str(item) for item in self._data)}]"

    __repr__ = __str__

    def to_tree_str(self) -> str:
        """
        Generates string representation of heap in Breadth First Ordering Format
        :return: String to print
        """
        string = ""

        # level spacing - init
        nodes_on_level = 0
        level_limit = 1
        spaces = 10 * int(1 + len(self))

        for i in range(len(self)):
            space = spaces // level_limit
            # determine spacing

            # add node to str and add spacing
            string += str(self._data[i]).center(space, ' ')

            # check if moving to next level
            nodes_on_level += 1
            if nodes_on_level == level_limit:
                string += '\n'
                level_limit *= 2
                nodes_on_level = 0
            i += 1

        return string

    def is_min_heap(self) -> bool:
        """
        Check if priority queue is a min or a max-heap
        :return: True if min-heap, False if max-heap
        """
        return self._is_min

    #   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   Modify below this line
    def __len__(self) -> int:
        """
        Determines the amount of nodes on the heap
        :return: The amount of nodes in the priority queue
        """
        return len(self._data)

    def empty(self) -> bool:
        """
        Checks if the heap is empty
        :return: bool - True if Empty, else False
        """
        return len(self) == 0

    def peek(self) -> PriorityNode:
        """
        Gets the root node (min or max node)
        :return: MinNode or MaxNode - None if heap is empty, else root node
        """
        if len(self) != 0:
            return self._data[0]
        return None

    def get_left_child_index(self, index: int) -> int:
        """
        Gets the specified parent node's left child index
        :param index: Index of parent node
        :return: Index of left child or None if it does not exist
        """
        left = 2 * index + 1
        if left < len(self._data):
            return 2 * index + 1
        return None

    def get_right_child_index(self, index: int) -> int:
        """
        Gets the specified parent node's right child index
        :param index: Index of parent node
        :return: Index of right child or None if it does not exist
        """
        right = 2 * index + 2
        if right < len(self._data):
            return 2 * index + 2
        return None

    def get_parent_index(self, index: int) -> int:
        """
        Gets the specified child node's parent index
        :param index: Index of child node
        :return: Index of parent or None if does not exist
        """
        if index == 0:
            return None
        return (index - 1) // 2

    def push(self, priority: Any, val: Any) -> None:
        """
        Inserts a node with the specified priority/value pair onto the heap
        :param priority: Node's priority
        :param val: Node's value
        :return: None
        """
        if self._is_min:
            self._data.append(MinNode(priority, val))
        elif not self._is_min:
            self._data.append(MaxNode(priority, val))
        self.percolate_up(len(self) - 1)

    def pop(self) -> PriorityNode:
        """
        Removes the top priority node from heap (min or max element)
        :return: MinNode or MaxNode - The root node of the heap
        """
        if len(self) == 0:
            return None
        result = self.peek()
        self._data[0] = self._data[-1]
        del self._data[-1]
        self.percolate_down(0)
        return result

    def get_minmax_child_index(self, index: int) -> int:
        """
        Gets the specified parent's min (min-heap) or max (max-heap) child index
        :param index: Index of parent element
        :return: Index of min child (if min-heap) or max child (if max-heap) or None if invalid
        """
        left = self.get_left_child_index(index)
        right = self.get_right_child_index(index)
        if left is None and right is None:
            return None
        elif left is None and right:
            return right
        elif right is None and left:
            return left
        elif left is not None and right is not None:
            if self._data[left] < self._data[right]:
                return left
            return right

    def percolate_up(self, index: int) -> None:
        """
        Moves a node in the queue/heap up to its correct position (level in the tree)
        :param index: Index of node to be percolated up
        :return: None
        """
        if index == 0:
            return
        parent = self.get_parent_index(index)
        if index > 0 and self._data[index] < self._data[parent]:
            self._data[parent], self._data[index] = self._data[index], self._data[parent]
            self.percolate_up(parent)

    def percolate_down(self, index: int) -> None:
        """
        Moves a node in the queue/heap down to its correct position (level in the tree)
        :param index: Index of node to be percolated down
        :return: None
        """
        parent = self.get_minmax_child_index(index)
        if parent is not None and self._data[index] > self._data[parent]:
            self._data[index], self._data[parent] = self._data[parent], self._data[index]
            self.percolate_down(parent)

class MaxHeap:
    """
    Implementation of a max-heap - the highest value is at the front (root).

    Initializes a PriorityQueue with is_min set to False.

    Uses the priority queue to satisfy the min heap properties by initializing
    the priority queue as a max-heap, and then using value as both the priority
    and value.
    """

    #   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   DO NOT MODIFY the following attributes/functions
    #   Modify only below indicated line

    __slots__ = ['_pqueue']

    def __init__(self):
        """
        Constructs a priority queue as a max-heap
        """
        self._pqueue = PriorityQueue(False)

    def __str__(self) -> str:
        """
        Represents the max-heap as a string
        :return: string representation of the heap
        """
        # NOTE: This hides implementation details
        return F"MaxHeap [{', '.join(item.value for item in self._pqueue._data)}]"

    __repr__ = __str__

    def to_tree_str(self) -> str:
        """
        Generates string representation of heap in Breadth First Ordering Format
        :return: String to print
        """
        return self._pqueue.to_tree_str()

    def __len__(self) -> int:
        """
        Determine the amount of nodes on the heap
        :return: Length of the data inside the heap
        """
        return len(self._pqueue)

    def empty(self) -> bool:
        """
        Checks if the heap is empty
        :returns: True if empty, else False
        """
        return self._pqueue.empty()

    #   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   Modify below this line
    def peek(self) -> Any:
        """
        Gets the max element's value (root node's value)
        :return: None if heap is empty, else root's value
        """
        return self._pqueue.peek().value

    def push(self, val: Any) -> None:
        """
        Inserts a node with the specified value onto the heap
        :param val: Node's value
        :return: None
        """
        self._pqueue.push(val, val)

    def pop(self) -> Any:
        """
        Removes the max element from the heap
        :return: Value of max element
        """
        if self.empty():
            return None
        return self._pqueue.pop().value


class MinHeap(MaxHeap):
    """
    Implementation of a max-heap - the highest value is at the front (root).

    Initializes a PriorityQueue with is_min set to True.

    Inherits from MaxHeap because it uses the same exact functions, but instead
    has a priority queue with a min-heap.
    """

    #   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   DO NOT MODIFY the following attributes/functions
    __slots__ = []

    def __init__(self):
        """
        Constructs a priority queue as a min-heap
        """
        super().__init__()
        self._pqueue._is_min = True


def heap_sort(array: List[Any]) -> None:
    """
    Sort array in-place using heap sort algorithm w/ max-heap
    :param array: List to be sorted
    :return: None
    """
    heap = MaxHeap()
    for i in array:
        heap.push(i)
    for j in range(len(heap)):
        array[j] = heap.pop()
    array.reverse()

def current_medians(array: List[int]) -> List[int]:
    """
    Calculate and record the median difficulty for each submission
    :param array: A list of numeric values
    :return: List of current medians in order data was read in
    """
    mins = MinHeap()
    maxs = MaxHeap()
    median = [0] * len(array)

    for i in range(len(array)):
        if not median:
            median[i] = array[i]
            mins.push(i)
        else:
            if maxs.empty() or array[i] < maxs.peek():
                maxs.push(array[i])
            else:
                mins.push(array[i])

            if len(mins) - len(maxs) > 1:
                maxs.push(mins.pop())
            elif len(maxs) - len(mins) > 1:
                mins.push(maxs.pop())

            if len(mins) == len(maxs):
                median[i] = (mins.peek() + maxs.peek()) / 2
            else:
                median[i] = maxs.peek() if len(maxs) > len(mins) else mins.peek()
    return median

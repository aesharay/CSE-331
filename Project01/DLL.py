"""
Project 1
CSE 331 S21 (Onsay)
Your Name
DLL.py
"""

from typing import TypeVar, List, Tuple
import datetime

# for more information on typehinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")            # represents generic type
Node = TypeVar("Node")      # represents a Node object (forward-declare to use in Node __init__)

# pro tip: PyCharm auto-renders docstrings (the multiline strings under each function definition)
# in its "Documentation" view when written in the format we use here. Open the "Documentation"
# view to quickly see what a function does by placing your cursor on it and using CTRL + Q.
# https://www.jetbrains.com/help/pycharm/documentation-tool-window.html


class Node:
    """
    Implementation of a doubly linked list node.
    Do not modify.
    """
    __slots__ = ["value", "next", "prev"]

    def __init__(self, value: T, next: Node = None, prev: Node = None) -> None:
        """
        Construct a doubly linked list node.

        :param value: value held by the Node.
        :param next: reference to the next Node in the linked list.
        :param prev: reference to the previous Node in the linked list.
        :return: None.
        """
        self.next = next
        self.prev = prev
        self.value = value

    def __repr__(self) -> str:
        """
        Represents the Node as a string.

        :return: string representation of the Node.
        """
        return str(self.value)

    def __str__(self) -> str:
        """
        Represents the Node as a string.

        :return: string representation of the Node.
        """
        return str(self.value)


class DLL:
    """
    Implementation of a doubly linked list without padding nodes.
    Modify only below indicated line.
    """
    __slots__ = ["head", "tail", "size"]

    def __init__(self) -> None:
        """
        Construct an empty doubly linked list.

        :return: None.
        """
        self.head = self.tail = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the DLL as a string.

        :return: string representation of the DLL.
        """
        result = ""
        node = self.head
        while node is not None:
            result += str(node)
            if node.next is not None:
                result += " <-> "
            node = node.next
        return result

    def __str__(self) -> str:
        """
        Represent the DLL as a string.

        :return: string representation of the DLL.
        """
        return repr(self)

    # MODIFY BELOW #

    def empty(self) -> bool:
        """
        Return boolean indicating whether DLL is empty.

        Suggested time & space complexity (respectively): O(1) & O(1).

        :return: True if DLL is empty, else False.
        """
        if self.size == 0:
            return True
        return False

    def push(self, val: T, back: bool = True) -> None:
        """
        Create Node containing `val` and add to back (or front) of DLL. Increment size by one.

        Suggested time & space complexity (respectively): O(1) & O(1).

        :param val: value to be added to the DLL.
        :param back: if True, add Node containing value to back (tail-end) of DLL;
            if False, add to front (head-end).
        :return: None.
        """
        val_node = Node(val)
        if self.size == 0:
            self.head = val_node
            self.tail = val_node
        elif back:
            self.tail.next = val_node
            val_node.prev = self.tail
            self.tail = val_node
        else:
            self.head.prev = val_node
            val_node.next = self.head
            self.head = val_node
        self.size = self.size + 1

    def pop(self, back: bool = True) -> None:
        """
        Remove Node from back (or front) of DLL. Decrement size by 1. If DLL is empty, do nothing.

        Suggested time & space complexity (respectively): O(1) & O(1).

        :param back: if True, remove Node from (tail-end) of DLL;
            if False, remove from front (head-end).
        :return: None.
        """
        if self.size == 0:
            return

        if back:
            temp_tail = self.tail.prev
            self.tail = temp_tail
            if self.size == 1:
                self.head = self.tail
            else:
                self.tail.next = None
        else:
            temp_head = self.head.next
            self.head = temp_head
            if self.size == 1:
                self.tail = self.head
            else:
                self.head.prev = None
        self.size = self.size - 1

    def from_list(self, source: List[T]) -> None:
        """
        Construct DLL from a standard Python list.

        Suggested time & space complexity (respectively): O(n) & O(n).

        :param source: standard Python list from which to construct DLL.
        :return: None.
        """
        for i in source:
            val_node = Node(i)
            if self.size == 0:
                self.head = val_node
                self.tail = val_node
            else:
                self.tail.next = val_node
                val_node.prev = self.tail
                self.tail = val_node
            self.size = self.size + 1

    def to_list(self) -> List[T]:
        """
        Construct standard Python list from DLL.

        Suggested time & space complexity (respectively): O(n) & O(n).

        :return: standard Python list containing values stored in DLL.
        """
        values = []
        first_head = self.head
        while first_head != None:
            values.append(first_head.value)
            first_head = first_head.next
        return values

    def find(self, val: T) -> Node:
        """
        Find first instance of `val` in the DLL and return associated Node object.

        Suggested time & space complexity (respectively): O(n) & O(1).

        :param val: value to be found in DLL.
        :return: first Node object in DLL containing `val`.
            If `val` does not exist in DLL, return None.
        """
        first_head = self.head
        while first_head is not None:
            if val == first_head.value:
                return first_head
            first_head = first_head.next
        return None


    def find_all(self, val: T) -> List[Node]:
        """
        Find all instances of `val` in DLL and return Node objects in standard Python list.

        Suggested time & space complexity (respectively): O(n) & O(n).

        :param val: value to be searched for in DLL.
        :return: Python list of all Node objects in DLL containing `val`.
            If `val` does not exist in DLL, return empty list.
        """
        values = []
        first_head = self.head
        while first_head is not None:
            if val == first_head.value:
                values.append(first_head)
            first_head = first_head.next
        return values

    def delete(self, val: T) -> bool:
        """
        Delete first instance of `val` in the DLL.

        Suggested time & space complexity (respectively): O(n) & O(1).

        :param val: value to be deleted from DLL.
        :return: True if Node containing `val` was deleted from DLL; else, False.
        """
        if self.head == None:
            return False
        elif self.size == 1:
            if self.head.value == val:
                temp_head = self.head.next
                self.head = temp_head
                self.tail = self.head
                self.size = self.size - 1
                return True
            return False
        elif self.head.value == val:
            temp_head = self.head.next
            self.head = temp_head
            self.head.prev = None
            self.size = self.size - 1
            return True
        elif self.tail.value == val:
            temp_tail = self.tail.prev
            self.tail = temp_tail
            self.tail.next = None
            self.size = self.size - 1
            return True

        temp_head = self.head
        while temp_head is not None:
            if temp_head.value == val:
                temp_next = temp_head.next
                temp_prev = temp_head.prev
                temp_prev.next = temp_next
                temp_next.prev = temp_prev
                self.size = self.size - 1
                return True
            temp_head = temp_head.next
        return False



    def delete_all(self, val: T) -> int:
        """
        Delete all instances of `val` in the DLL.

        Suggested time & space complexity (respectively): O(n) & O(1).

        :param val: value to be deleted from DLL.
        :return: integer indicating the number of Nodes containing `val` deleted from DLL;
                 if no Node containing `val` exists in DLL, return 0.
        """
        count = 0
        if self.head == None:
            return count
        if self.size == 1:
            if self.head.value == val:
                self.delete(val)
                return 1
            return 0

        temp = self.head
        while temp is not None:
            flag = self.delete(val)
            if flag:
                count += 1
            temp = temp.next
        return count


    def reverse(self) -> None:
        """
        Reverse DLL in-place by modifying all `next` and `prev` references of Nodes in the
        DLL and resetting the `head` and `tail` references.
        Must be implemented in-place for full credit. May not create new Node objects.

        Suggested time & space complexity (respectively): O(n) & O(1).

        :return: None.
        """
        if self.head is None:
            return

        temp = self.head
        curr = self.head
        while curr:
            temp = curr.prev
            curr.prev = curr.next
            curr.next = temp
            curr = curr.prev
        self.head, self.tail = self.tail, self.head

class Stock:
    """
    Implementation of a stock price on a given day.
    Do not modify.
    """

    __slots__ = ["date", "price"]

    def __init__(self, date: datetime.date, price: float) -> None:
        """
        Construct a stock.

        :param date: date of stock.
        :param price: the price of the stock at the given date.
        """
        self.date = date
        self.price = price

    def __repr__(self) -> str:
        """
        Represents the Stock as a string.

        :return: string representation of the Stock.
        """
        return f"<{str(self.date)}, ${self.price}>"

    def __str__(self) -> str:
        """
        Represents the Stock as a string.

        :return: string representation of the Stock.
        """
        return repr(self)


def intellivest(stocks: DLL) -> Tuple[datetime.date, datetime.date, float]:
    """
    Given a DLL representing daily stock prices,
    find the optimal streak of days over which to invest.
    To be optimal, the streak of stock prices must:

        (1) Be strictly increasing, such that the price of the stock on day i+1
        is greater than the price of the stock on day i, and
        (2) Have the greatest total increase in stock price from
        the first day of the streak to the last.

    In other words, the optimal streak of days over which to invest is the one over which stock
    price increases by the greatest amount, without ever going down (or staying constant).

    Suggested time & space complexity (respectively): O(n) & O(1).

    :param stocks: DLL with Stock objects as node values, as defined above.
    :return: Tuple with the following elements:
        [0]: date: The date at which the optimal streak begins.
        [1]: date: The date at which the optimal streak ends.
        [2]: float: The (positive) change in stock price between the start and end
                dates of the streak.
    """
    nodes = list()
    nodes = stocks.to_list()
    increase = 0
    low_price = 0
    low_date = ""
    high_price = 0
    high_date = ""

    if len(nodes) == 0:
        return None, None, 0
    elif len(nodes) == 1:
        return nodes[0].date, nodes[0].date, 0
    elif len(nodes) == 2:
        if nodes[0].price > nodes[1].price:
            low_date = nodes[0].date
            high_date = nodes[0].date
        elif nodes[0].price < nodes[1].price:
            low_price = nodes[0].price
            low_date = nodes[0].date
            high_price = nodes[1].price
            high_date = nodes[1].date
        elif nodes[0].price == nodes[1].price:
            return nodes[0].date, nodes[0].date, 0
        increase = high_price - low_price
        return low_date, high_date, increase

    temp = nodes[0]
    inc_lst = list()

    for i in range(1, len(nodes)-1):
        if temp.price < nodes[i].price:
            low_price = temp.price
            low_date = temp.date
            high_price = nodes[i].price
            high_date = nodes[i + 1].date
            temp = nodes[i]
            inc_lst.append((low_date, high_date, increase))
        elif temp.price > nodes[i].price:
            low_price = nodes[i].price
            low_date = nodes[i].date
            high_price = temp.price
            high_date = temp.date
            temp = nodes[i]
        else:
            low_price = high_price = temp.price
            low_date = high_date = temp.date
        increase = high_price - low_price

    inc_tup = inc_lst[0]
    for i in range(1,len(inc_lst)-1):
        if inc_lst[i][2] > inc_lst[i - 1][2]:
            inc_tup = inc_lst[i]
    return inc_tup

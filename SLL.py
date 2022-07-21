"""
Project 1
CSE 331 S21 (Onsay)
Your Name
DLL.py
"""

from Node import Node       # Import `Node` class
from typing import TypeVar  # For use in type hinting

# Type Declarations
T = TypeVar('T')        # generic type
SLL = TypeVar('SLL')    # forward declared


class RecursiveSinglyLinkList:
    """
    Recursive implementation of an SLL
    """

    __slots__ = ['head']

    def __init__(self) -> None:
        """
        Initializes an `SLL`
        :return: None
        """
        self.head = None

    def __repr__(self) -> str:
        """
        Represents an `SLL` as a string
        """
        return self.to_string(self.head)

    def __str__(self) -> str:
        """
        Represents an `SLL` as a string
        """
        return self.to_string(self.head)

    def __eq__(self, other: SLL) -> bool:
        """
        Overloads `==` operator to compare SLLs
        :param other: right hand operand of `==`
        :return: `True` if equal, else `False`
        """
        comp = lambda n1, n2: n1 == n2 and (comp(n1.next, n2.next) if (n1 and n2) else True)
        return comp(self.head, other.head)

# ============ Modify below ============ #

    def to_string(self, curr: Node) -> str:
        """
        Transform list and nodes into a string
        If list is empty, return None
        """
        if self.head is None:
            return "None"
        if curr is self.head:
            result = str(curr.val)
            curr = curr.next
        else:
            arrow = " --> "
            result = arrow
            result += str(curr.val)
            curr = curr.next
        if curr is None:
            return result
        result += self.to_string(curr)
        return result

    def length(self, curr: Node) -> int:
        """
        Return number of Nodes in the list
        """
        if not curr:
            return 0
        return 1 + self.length(curr.next)

    def sum_list(self, curr: Node) -> T:
        """
        Calculates and returns the sum of the values in the list
        """
        if curr is not None:
            return curr.val + self.sum_list(curr.next)
        return 0

    def push(self, value: T) -> None:
        """
        Insert given value into the list
        """
        def push_inner(curr: Node) -> None:
            """
            Insert value from push at the end of the list
            """
            """temp = curr
            if temp.next is None:
                temp.next = Node(value)
                return temp.next
            temp = temp.next
            push_inner(temp)"""
            if curr.next is not None:
                return push_inner(curr.next)
            else:
                curr.next = Node(value)

        push_inner(self.head)
        """if self.head is None:
            self.head = Node(value)
            return self.head
        push_inner(self.head)"""

    def remove(self, value: T) -> None:
        """
        Remove the first node with the given value
        If list is empty, keep the list the same
        """
        def remove_inner(curr: Node) -> Node:
            """
            Find and remove the first node with the given value
            in the list
            If value is not in list, keep the original list
            """
            temp = curr
            if temp.next is None:
                return None
            if temp.next.val == value:
                temp.next = temp.next.next
                return None
            remove_inner(temp.next)

        if self.head is None:
            return None
        if self.head.val == value:
            self.head = self.head.next
            return None
        remove_inner(self.head)

    def remove_all(self, value: T) -> None:
        """
        Remove all instances of the given value in the list
        If value doesn't exist, keep the original lsit
        """
        def remove_all_inner(curr):
            """
            Helper function for remove
            """
            temp = curr
            if temp.next is None:
                return None
            if temp.next.val == value:
                if temp.next == self.head:
                    self.head = temp.next.next
                temp.next = temp.next.next
            remove_all_inner(temp.next)

        curr = self.head
        if self.head is None:
            return None
        if self.head.val == value:
            self.head = self.head.next
        remove_all_inner(curr)

    def search(self, value: T) -> bool:
        """
        Search for given value
        """
        def search_inner(curr):
            """
            Helper function to search for value
            If found, return True
            Else, return False
            """
            temp = curr
            if temp is None:
                return False
            if temp.val == value:
                return True
            return search_inner(temp.next)

        if self.head is None:
            return False
        result = search_inner(self.head)
        return result

    def count(self, value: T) -> int:
        """
        Count number of nodes with the given value
        """
        def count_inner(curr):
            """
            Helper function to count for nodes
            If node with given value is found, increase count
            Else, return 0
            """
            temp = curr
            if temp is None:
                return 0
            if temp.val == value:
                return 1 + count_inner(temp.next)
            return count_inner(temp.next)

        if self.head is None:
            return 0
        result = count_inner(self.head)
        return result

    def reverse(self, curr):
        """
        Reverse list
        If list is empty, return head
        """
        if curr is None:
            return None
        if curr.next is None:
            self.head = curr
            return curr
        self.reverse(curr.next)
        curr.next.next = curr
        curr.next = None

def crafting(recipe, pockets):
    """
        Build new item with ingredients in pockets and recipe
        If pockets contain all items in the recipe, return True
        Edit pocket and recipe with new inventory
        If pockets do not contain all items needed, return False
        Keep original pocket and recipe list
        """
    rec_item = recipe.head
    pock_item = pockets.head
    if rec_item is None or pock_item is None:
        return False

    def crafting_inner(rec_item):
        """
        Check to see if given item is found in pockets
        If found, return true
        If not found, return false
        """
        result = True
        count_recipe = recipe.count(rec_item.val)
        count_pockets = pockets.count(rec_item.val)
        if count_pockets >= count_recipe:
            if rec_item.next is None:
                return result
            rec_item = rec_item.next
            return crafting_inner(rec_item)
        result = False
        return result

    def remove_item(rec_item):
        """
        Function removes item that is used in recipe
        Updates pocket list
        """
        pockets.remove(rec_item.val)
        if rec_item.next is None:
            return None
        rec_item = rec_item.next
        remove_item(rec_item)

    result = crafting_inner(rec_item)
    if result:
        remove_item(rec_item)
    return result

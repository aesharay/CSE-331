"""
Project 5
CSE 331 S21 (Onsay)
Your Name
AVLTree.py
"""

import queue
from typing import TypeVar, Generator, List, Tuple
from queue import Queue

# for more information on typehinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")            # represents generic type
Node = TypeVar("Node")      # represents a Node object (forward-declare to use in Node __init__)
AVLWrappedDictionary = TypeVar("AVLWrappedDictionary")      # represents a custom type used in application


####################################################################################################


class Node:
    """
    Implementation of an AVL tree node.
    Do not modify.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["value", "parent", "left", "right", "height"]

    def __init__(self, value: T, parent: Node = None,
                 left: Node = None, right: Node = None) -> None:
        """
        Construct an AVL tree node.

        :param value: value held by the node object
        :param parent: ref to parent node of which this node is a child
        :param left: ref to left child node of this node
        :param right: ref to right child node of this node
        """
        self.value = value
        self.parent, self.left, self.right = parent, left, right
        self.height = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return f"<{str(self.value)}>"

    def __str__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return f"<{str(self.value)}>"


####################################################################################################


class AVLTree:
    """
    Implementation of an AVL tree.
    Modify only below indicated line.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty AVL tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree as a string. Inspired by Anna De Biasi (Fall'20 Lead TA).

        :return: string representation of the AVL tree
        """
        if self.origin is None:
            return "Empty AVL Tree"

        # initialize helpers for tree traversal
        root = self.origin
        result = ""
        q = queue.SimpleQueue()
        levels = {}
        q.put((root, 0, root.parent))
        for i in range(self.origin.height + 1):
            levels[i] = []

        # traverse tree to get node representations
        while not q.empty():
            node, level, parent = q.get()
            if level > self.origin.height:
                break
            levels[level].append((node, level, parent))

            if node is None:
                q.put((None, level + 1, None))
                q.put((None, level + 1, None))
                continue

            if node.left:
                q.put((node.left, level + 1, node))
            else:
                q.put((None, level + 1, None))

            if node.right:
                q.put((node.right, level + 1, node))
            else:
                q.put((None, level + 1, None))

        # construct tree using traversal
        spaces = pow(2, self.origin.height) * 12
        result += "\n"
        result += f"AVL Tree: size = {self.size}, height = {self.origin.height}".center(spaces)
        result += "\n\n"
        for i in range(self.origin.height + 1):
            result += f"Level {i}: "
            for node, level, parent in levels[i]:
                level = pow(2, i)
                space = int(round(spaces / level))
                if node is None:
                    result += " " * space
                    continue
                if not isinstance(self.origin.value, AVLWrappedDictionary):
                    result += f"{node} ({parent} {node.height})".center(space, " ")
                else:
                    result += f"{node}".center(space, " ")
            result += "\n"
        return result

    def __str__(self) -> str:
        """
        Represent the AVL tree as a string. Inspired by Anna De Biasi (Fall'20 Lead TA).

        :return: string representation of the AVL tree
        """
        return repr(self)

    def height(self, root: Node) -> int:
        """
        Return height of a subtree in the AVL tree, properly handling the case of root = None.
        Recall that the height of an empty subtree is -1.

        :param root: root node of subtree to be measured
        :return: height of subtree rooted at `root` parameter
        """
        return root.height if root is not None else -1

    def left_rotate(self, root: Node) -> Node:
        """
        Perform a left rotation on the subtree rooted at `root`. Return new subtree root.

        :param root: root node of unbalanced subtree to be rotated.
        :return: new root node of subtree following rotation.
        """
        if root is None:
            return None

        # pull right child up and shift right-left child across tree, update parent
        new_root, rl_child = root.right, root.right.left
        root.right = rl_child
        if rl_child is not None:
            rl_child.parent = root

        # right child has been pulled up to new root -> push old root down left, update parent
        new_root.left = root
        new_root.parent = root.parent
        if root.parent is not None:
            if root is root.parent.left:
                root.parent.left = new_root
            else:
                root.parent.right = new_root
        root.parent = new_root

        # handle tree origin case
        if root is self.origin:
            self.origin = new_root

        # update heights and return new root of subtree
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        new_root.height = 1 + max(self.height(new_root.left), self.height(new_root.right))
        return new_root

    ########################################
    # Implement functions below this line. #
    ########################################

    def right_rotate(self, root: Node) -> Node:
        """
        Perform a right rotation on the subtree rooted at root.
        Return root of new subtree after rotation.
        """
        if root is None:
            return None

            # pull right child up and shift right-left child across tree, update parent
        new_root, rl_child = root.left, root.left.right
        root.left = rl_child
        if rl_child is not None:
            rl_child.parent = root

        # right child has been pulled up to new root -> push old root down left, update parent
        new_root.right = root
        new_root.parent = root.parent
        if root.parent is not None:
            if root is root.parent.right:
                root.parent.right = new_root
            else:
                root.parent.left = new_root
        root.parent = new_root

        # handle tree origin case
        if root is self.origin:
            self.origin = new_root

        # update heights and return new root of subtree
        root.height = 1 + max(self.height(root.right), self.height(root.left))
        new_root.height = 1 + max(self.height(new_root.right), self.height(new_root.left))
        return new_root

    def balance_factor(self, root: Node) -> int:
        """
        Compute the balance factor of the subtree rooted at root.
        """
        if root is None:
            return 0
        return self.height(root.left) - self.height(root.right)

    def rebalance(self, root: Node) -> Node:
        """
        Rebalance the subtree rooted at root and return the new root of the resulting subtree.
        """
        balance = self.balance_factor(root)
        if balance == 2:
            if self.balance_factor(root.left) == -1:
                self.left_rotate(root.left)
            return self.right_rotate(root)
        if balance == -2:
            if self.balance_factor(root.right) == 1:
                self.right_rotate(root.right)
            return self.left_rotate(root)
        return root

    def insert(self, root: Node, val: T) -> Node:
        """
        REPLACE
        """
        if self.origin is None:
            self.origin = Node(val)
        if root is None:
            root = Node(val)
            self.size += 1
        if val > root.value:
            right_root = self.insert(root.right, val)
            root.right = right_root
            right_root.parent = root
        if val < root.value:
            left_root = self.insert(root.left, val)
            root.left = left_root
            left_root.parent = root
        root.height = 1 + (max(self.height(root.left), self.height(root.right)))
        return self.rebalance(root)

    def min(self, root: Node) -> Node:
        """
        Find and return the Node with the smallest value in the subtree rooted at root
        """
        if self.origin is None:
            return None
        if root is None:
            return None
        if root.left is not None:
            min_val = self.min(root.left)
        else:
            min_val = root
        return min_val

    def max(self, root: Node) -> Node:
        """
        Find and return the Node with the largest value in the subtree rooted at root
        """
        if self.origin is None:
            return None
        if root is None:
            return None
        if root.right is not None:
            max_val = self.max(root.right)
        else:
            max_val = root
        return max_val

    def search(self, root: Node, val: T) -> Node:
        """
        Find and return the Node with the value val in the subtree rooted at root
        """
        temp = None
        while root is not None:
            temp = root
            if root.value < val:
                root = root.right
            elif root.value > val:
                root = root.left
            elif root.value == val:
                break
        return temp

    def inorder(self, root: Node) -> Generator[Node, None, None]:
        """"
        Perform an inorder (left, current, right) traversal of the subtree rooted at root
        """
        if root is None:
            return
        yield from self.inorder(root.left)
        yield root
        yield from self.inorder(root.right)

    def preorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Perform an inorder (left, current, right) traversal of the subtree rooted at root
        """
        if root is None:
            return
        yield root
        yield from self.preorder(root.left)
        yield from self.preorder(root.right)

    def postorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Perform a postorder (left, right, current) traversal of the subtree rooted at root
        """
        if root is None:
            return
        yield from self.postorder(root.left)
        yield from self.postorder(root.right)
        yield root

    def levelorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Function performs a level-order traversal of the subtree rooted at root
        Returns generator object which yields None objects only
        """
        if root is None:
            return
        q = queue.SimpleQueue()
        q.put(root)
        while not q.empty():
            node = q.get()
            yield node
            if node.left:
                q.put(node.left)
            if node.right:
                q.put(node.right)

    def remove(self, root: Node, val: T) -> Node:
        """
        Remove the node with value val from the subtree rooted at root
        Return the root of the balanced subtree following removal
        If val does not exist in the AVL tree, return None
        """
        if root is None:
            return None
        elif val < root.value:
            self.remove(root.left, val)
        elif val > root.value:
            self.remove(root.right, val)
        else:
            if root.right is None and root.left is None:
                par = root.parent
                if root is self.origin:
                    self.origin = None
                elif par.right is not None and par.right.value == val:
                    par.right = None
                else:
                    par.left = None
                self.size -= 1
            elif root.right is None or root.left is None:
                par = root.parent
                if root.right is not None:
                    child = root.right
                else:
                    child = root.left
                if root is self.origin:
                    self.origin = child
                elif par.right is not None and par.right.value == val:
                    par.right = child
                else:
                    par.left = child
                child.parent = par
                self.size -= 1
            else:
                max_left = self.max(root.left)
                root.value = max_left.value
                self.remove(root.left, root.value)
        root.height = 1 + (max(self.height(root.left), self.height(root.right)))
        return self.rebalance(root)

####################################################################################################


class AVLWrappedDictionary:
    """
    Implementation of a helper class which will be used as tree node values in the
    NearestNeighborClassifier implementation. Compares objects with keys less than
    1e-6 apart as equal.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["key", "dictionary"]

    def __init__(self, key: float) -> None:
        """
        Construct a AVLWrappedDictionary with a key to search/sort on and a dictionary to hold data.

        :param key: floating point key to be looked up by.
        """
        self.key = key
        self.dictionary = {}

    def __repr__(self) -> str:
        """
        Represent the AVLWrappedDictionary as a string.

        :return: string representation of the AVLWrappedDictionary.
        """
        return f"key: {self.key}, dict: {self.dictionary}"

    def __str__(self) -> str:
        """
        Represent the AVLWrappedDictionary as a string.

        :return: string representation of the AVLWrappedDictionary.
        """
        return f"key: {self.key}, dict: {self.dictionary}"

    def __eq__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement == operator to compare 2 AVLWrappedDictionaries by key only.
        Compares objects with keys less than 1e-6 apart as equal.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating whether keys of AVLWrappedDictionaries are equal
        """
        return abs(self.key - other.key) < 1e-6

    def __lt__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement < operator to compare 2 AVLWrappedDictionarys by key only.
        Compares objects with keys less than 1e-6 apart as equal.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating ordering of AVLWrappedDictionaries
        """
        return self.key < other.key and not abs(self.key - other.key) < 1e-6

    def __gt__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement > operator to compare 2 AVLWrappedDictionaries by key only.
        Compares objects with keys less than 1e-6 apart as equal.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating ordering of AVLWrappedDictionaries
        """
        return self.key > other.key and not abs(self.key - other.key) < 1e-6


class NearestNeighborClassifier:
    """
    Implementation of a one-dimensional nearest-neighbor classifier with AVL tree lookups.
    Modify only below indicated line.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["resolution", "tree"]

    def __init__(self, resolution: int) -> None:
        """
        Construct a one-dimensional nearest neighbor classifier with AVL tree lookups.
        Data are assumed to be floating point values in the closed interval [0, 1].

        :param resolution: number of decimal places the data will be rounded to, effectively
                           governing the capacity of the model - for example, with a resolution of
                           1, the classifier could maintain up to 11 nodes, spaced 0.1 apart - with
                           a resolution of 2, the classifier could maintain 101 nodes, spaced 0.01
                           apart, and so on - the maximum number of nodes is bounded by
                           10^(resolution) + 1.
        """
        self.tree = AVLTree()
        self.resolution = resolution

        # pre-construct lookup tree with AVLWrappedDictionary objects storing (key, dictionary)
        # pairs, but which compare with <, >, == on key only
        for i in range(10**resolution + 1):
            w_dict = AVLWrappedDictionary(key=(i/10**resolution))
            self.tree.insert(self.tree.origin, w_dict)

    def __repr__(self) -> str:
        """
        Represent the NearestNeighborClassifier as a string.

        :return: string representation of the NearestNeighborClassifier.
        """
        return f"NNC(resolution={self.resolution}):\n{self.tree}"

    def __str__(self) -> str:
        """
        Represent the NearestNeighborClassifier as a string.

        :return: string representation of the NearestNeighborClassifier.
        """
        return f"NNC(resolution={self.resolution}):\n{self.tree}"

    def fit(self, data: List[Tuple[float, str]]) -> None:
        """
        Function to learn the associations between features x and target labels y
        """
        for x, y in data:
            x = round(x, self.resolution)
            root = self.tree.search(self.tree.origin, AVLWrappedDictionary(x))
            if y in root.value.dictionary:
                root.value.dictionary[y] += 1
            root.value.dictionary[y] = 1

    def predict(self, x: float, delta: float) -> str:
        """
        Function that predicts the unknown labels of features x
        """
        x = round(x, self.resolution)
        mini = x - delta
        maxa = x + delta
        result = {}

        def predict_helper(root):
            if mini <= root.value.key <= maxa:
                for k, v in root.value.dictionary.items():
                    if k in result:
                        result[k] += v
                    else:
                        result[k] = v
            if root.left is not None and root.value.key >= mini:
                predict_helper(root.left)
            if root.right is not None and root.value.key <= maxa:
                predict_helper(root.right)
        predict_helper(self.tree.origin)
        return max(result, key=result.get) if result else None

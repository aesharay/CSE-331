"""
Project 6
CSE 331 S21 (Onsay)
Your Name
hashtable.py
"""

from typing import TypeVar, List, Tuple

T = TypeVar("T")
HashNode = TypeVar("HashNode")
HashTable = TypeVar("HashTable")


class HashNode:
    """
    DO NOT EDIT
    """
    __slots__ = ["key", "value", "deleted"]

    def __init__(self, key: str, value: T, deleted: bool = False) -> None:
        self.key = key
        self.value = value
        self.deleted = deleted

    def __str__(self) -> str:
        return f"HashNode({self.key}, {self.value})"

    __repr__ = __str__

    def __eq__(self, other: HashNode) -> bool:
        return self.key == other.key and self.value == other.value

    def __iadd__(self, other: T) -> None:
        self.value += other


class HashTable:
    """
    Hash Table Class
    """
    __slots__ = ['capacity', 'size', 'table', 'prime_index']

    primes = (
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
        89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
        181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277,
        281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389,
        397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499,
        503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617,
        619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739,
        743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859,
        863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991,
        997)

    def __init__(self, capacity: int = 8) -> None:
        """
        DO NOT EDIT
        Initializes hash table
        :param capacity: capacity of the hash table
        """
        self.capacity = capacity
        self.size = 0
        self.table = [None] * capacity

        i = 0
        while HashTable.primes[i] <= self.capacity:
            i += 1
        self.prime_index = i - 1

    def __eq__(self, other: HashTable) -> bool:
        """
        DO NOT EDIT
        Equality operator
        :param other: other hash table we are comparing with this one
        :return: bool if equal or not
        """
        if self.capacity != other.capacity or self.size != other.size:
            return False
        for i in range(self.capacity):
            if self.table[i] != other.table[i]:
                return False
        return True

    def __str__(self) -> str:
        """
        DO NOT EDIT
        Represents the table as a string
        :return: string representation of the hash table
        """
        represent = ""
        bin_no = 0
        for item in self.table:
            represent += "[" + str(bin_no) + "]: " + str(item) + '\n'
            bin_no += 1
        return represent

    __repr__ = __str__

    def _hash_1(self, key: str) -> int:
        """
        ---DO NOT EDIT---
        Converts a string x into a bin number for our hash table
        :param key: key to be hashed
        :return: bin number to insert hash item at in our table, None if key is an empty string
        """
        if not key:
            return None
        hashed_value = 0

        for char in key:
            hashed_value = 181 * hashed_value + ord(char)
        return hashed_value % self.capacity

    def _hash_2(self, key: str) -> int:
        """
        ---DO NOT EDIT---
        Converts a string x into a hash
        :param key: key to be hashed
        :return: a hashed value
        """
        if not key:
            return None
        hashed_value = 0

        for char in key:
            hashed_value = 181 * hashed_value + ord(char)

        prime = HashTable.primes[self.prime_index]

        hashed_value = prime - (hashed_value % prime)
        if hashed_value % 2 == 0:
            hashed_value += 1
        return hashed_value

    def __len__(self) -> int:
        """
        Returns the size of the HashTable
        """
        return self.size

    def __setitem__(self, key: str, value: T) -> None:
        """
        Sets the value with an associated key in the HashTable
        """
        self._insert(key, value)

    def __getitem__(self, key: str) -> T:
        """
        Looks up the value with an associated key in the HashTable
        """
        if not self._get(key):
            raise KeyError
        return self._get(key).value

    def __delitem__(self, key: str) -> None:
        """
        Deletes the value with an associated key in the HashTable
        """
        if not self._get(key):
            raise KeyError
        return self._delete(key)

    def __contains__(self, key: str) -> bool:
        """
        Determines if a node with the key denoted by the parameter exists in the table
        """
        if self._get(key):
            return True

    def hash(self, key: str, inserting: bool = False) -> int:
        """
        Return the index with the given key string
        If the key does not exist, return the next available empty index
        """
        index = self._hash_1(key)
        if index is None:
            return 0
        i = 1
        node = self.table[index]
        while node:
            if not node.deleted:
                if node.key == key:
                    return index
            if inserting:
                if node.deleted:
                    return index
            index = (self._hash_1(key) + i * self._hash_2(key)) % self.capacity
            i += 1
            node = self.table[index]
        return index

    def _insert(self, key: str, value: T) -> None:
        """
        Inserts the given key and value into a HashNode
        Insert the HashNode into the Hashtable
        If the key does not exist, return the next available empty index
        """
        index = self.hash(key, inserting=True)
        if self.table[index] is None:
            self.table[index] = HashNode(key, value)
            self.size += 1
        if self.table[index].key is None:
            self.table[index].key = key
            self.table[index].value = value
            self.size += 1
        self.table[index].value = value
        increase = self.size / self.capacity
        if increase >= 0.5:
            self._grow()

    def _get(self, key: str) -> HashNode:
        """
        Find the HashNode with the given key in the hash table
        If the HashNode does not exist, return None
        """
        index = self.hash(key)
        if index is None:
            return None
        return self.table[index]

    def _delete(self, key: str) -> None:
        """
        Removes the HashNode with the given key from the hash table
        If the node is found assign its key and value to None
        Set the deleted flag to True
        """
        index = self.hash(key)
        if index is None:
            return
        node = self.table[index]
        if node:
            if node.key == key:
                node.key = None
                node.value = None
                node.deleted = True
                self.size -= 1
        return

    def _grow(self) -> None:
        """
        Double the capacity of the existing hash table
        """
        self.capacity *= 2
        new_hash = HashTable(self.capacity)
        self.prime_index = new_hash.prime_index
        index = 0
        while index < (self.capacity / 2):
            node = self.table[index]
            if node:
                if not node.deleted:
                    new_hash._insert(node.key, node.value)
            index += 1
        self.table = new_hash.table

    def update(self, pairs: List[Tuple[str, T]] = []) -> None:
        """
        Updates the hash table using an iterable of key value pairs
        """
        for item in pairs:
            if self._get(item[0]):
                self._get(item[0]).value = item[1]
            self._insert(item[0], item[1])

    def keys(self) -> List[str]:
        """
        Makes a list that contains all of the keys in the table
        """
        result = []
        for item in self.table:
            if item is not None:
                result.append(item.key)
        return result

    def values(self) -> List[T]:
        """
        Makes a list that contains all of the values in the table
        """
        result = []
        for item in self.table:
            if item is not None:
                result.append(item.value)
        return result

    def items(self) -> List[Tuple[str, T]]:
        """
        Makes a list that contains all of the items in the table
        """
        result = []
        for item in self.table:
            if item is not None:
                result.append((item.key, item.value))
        return result

    def clear(self) -> None:
        """
        Clears the table of HashNodes completely
        """
        self.size = 0
        self.table = [None] * self.capacity

class CataData:
    """
    CataData Class
    """
    def __init__(self) -> None:
        """
        Data structure with two HashTables
        start HashTable stores enter time
        avg HashTable stores number of trips
        """
        self.start = HashTable()
        self.avg = HashTable()

    def enter(self, idx: str, origin: str, time: int) -> None:
        """
        Stores passenger with enter time
        """
        self.start[idx] = (origin, time)

    def exit(self, idx: str, dest: str, time: int) -> None:
        """
        Updates when passenger exits the bus
        Stores exit time
        """
        if idx in self.start:
            start, ent_time = self.start[idx]
            if start not in self.avg:
                self.avg[start] = HashTable()
            if dest not in self.avg[start]:
                self.avg[start][dest] = (1, time - ent_time)
            else:
                people, total_time = self.avg[start][dest]
                self.avg[start][dest] = (people + 1, total_time + time - ent_time)

    def get_average(self, origin: str, dest: str) -> float:
        """
        Gets average travel time of passengers
        """
        if origin in self.avg:
            if dest in self.avg[origin]:
                people, total_time = self.avg[origin][dest]
                average = total_time / people
                return average
        return 0
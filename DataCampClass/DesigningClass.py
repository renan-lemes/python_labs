# %%
from datetime import datetime
import pandas as pd


class Parent:
    def talk(self):
        print("Parent talking!")


class Child(Parent):
    def talk(self):
        print("Child talking!")


class TalkativeChild(Parent):
    def talk(self):
        print("TalkativeChild talking!")
        Parent.talk(self)


p, c, tc = Parent(), Child(), TalkativeChild()

for obj in (p, c, tc):
    obj.talk()

# %%
# Define a Rectangle class


class Rectangle:
    def __init__(self, h, w):
        self.h, self.w = h, w

# Define a Square class


class Square(Rectangle):
    def __init__(self, w):
        self.h, self.w = w, w

# %%


class Rectangle:
    def __init__(self, w, h):
        self.w, self.h = w, h

# Define set_h to set h
    def set_h(self, h):
        self.h = h

# Define set_w to set w
    def set_w(self, w):
        self.w = w


class Square(Rectangle):
    def __init__(self, w):
        self.w, self.h = w, w

# Define set_h to set w and h
    def set_h(self, h):
        self.h = h
        self.w = h

# Define set_w to set w and h
    def set_w(self, w):
        self.w = w
        self.h = w

# %%


class Customer:
    def __init__(self, name, new_bal):
        self.name = name
        if new_bal < 0:
            raise ValueError("Invalid balance!")
        self._balance = new_bal

    # Add a decorated balance() method returning _balance
    @property
    def balance(self):
        return self._balance

    # Add a setter balance() method
    @balance.setter
    def balance(self, new_bal):
        # Validate the parameter value
        if new_bal < 0:
            raise ValueError("Invalid balance!")
        self._balance = new_bal
        print("Setter method called")


# Create a Customer
cust = Customer('Belinda Lutz', 2000)

# Assign 3000 to the balance property
cust.balance = 3000

# Print the balance property
print(cust.balance)

# %%


class LoggedDF(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        pd.DataFrame.__init__(self, *args, **kwargs)
        self._created_at = datetime.today()

    def to_csv(self, *args, **kwargs):
        temp = self.copy()
        temp["created_at"] = self._created_at
        pd.DataFrame.to_csv(temp, *args, **kwargs)

    # Add a read-only property: _created_at
    @property
    def created_at(self):
        return self._created_at


# Instantiate a LoggedDF called ldf
ldf = LoggedDF({"col1": [1, 2], "col2": [3, 4]})

ldf.created_at = '2020-01-01'
# %%



# %%


class BankAccount:
    # MODIFY to initialize a number attribute
    def __init__(self, balance=0, number=0):
        self.balance = balance
        self.number = number

    def withdraw(self, amount):
        self.balance -= amount

    # Define __eq__ that returns True if the number attributes are equal
    def __eq__(self, other):
        return other.balance == self.balance  # self.number == other.number


# Create accounts and compare them
acct1 = BankAccount(123, 1000)
acct2 = BankAccount(123, 1000)
acct3 = BankAccount(456, 1000)
print(acct1 == acct2)
print(acct1 == acct3)


# %%


class Parent:
    def __eq__(self, other):
        print("Parent's __eq__() called")
        return True


class Child(Parent):
    def __eq__(self, other):
        print("Child's __eq__() called")
        return True


p = Parent()
c = Child()

p == c
# %%
my_num = 5
my_str = "Hello"

#f = f = "my_num is {0}, and my_str is {1}.".format(my_num, my_str)
f = "my_num is {}, and my_str is \"{}\".".format(my_num, my_str)
#f = "my_num is {my_num}, and my_str is '{my_str}'.".format()
#f= "my_num is {n}, and my_str is '{s}'.".format(n=my_num, s=my_str)
print(f)

# %%


class Employee:
    def __init__(self, name, salary=30000):
        self.name, self.salary = name, salary

    # Add the __str__() method
    def __str__(self):
        s = "Employee name: {name}\nEmployee salary: {salary}".format(
            name=self.name, salary=self.salary)
        return s


emp1 = Employee("Amar Howard", 30000)
print(emp1)
emp2 = Employee("Carolyn Ramirez", 35000)
print(emp2)

# %%

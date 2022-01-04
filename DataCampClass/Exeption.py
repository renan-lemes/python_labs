# %%
def invert_at_index(x, ind):
    try:
        return 1/x[ind]
    except ZeroDivisionError:
        print("Cannot divide by zero")
    except IndexError:
        print("Index out of range")


a = [5, 6, 0, 7]

# Works okay
print(invert_at_index(a, 1))

# Potential ZeroDivisionError
print(invert_at_index(a, 2))

# Potential IndexError
print(invert_at_index(a, 5))


# %%
class SalaryError(ValueError):
    pass


class BonusError(SalaryError):
    pass


class Employee:
    MIN_SALARY = 30000
    MAX_RAISE = 5000

    def __init__(self, name, salary=30000):
        self.name = name

        # If salary is too low
        if MIN_SALARY >= self.salary:
            # Raise a SalaryError exception
            print("Salary is too low!")

        self.salary = salary

# %%


class SalaryError(ValueError):
    pass


class BonusError(SalaryError):
    pass


class Employee:
    MIN_SALARY = 30000
    MAX_BONUS = 5000

    def __init__(self, name, salary=30000):
        self.name = name
        if salary < Employee.MIN_SALARY:
            raise SalaryError("Salary is too low!")
        self.salary = salary

    # Rewrite using exceptions
    def give_bonus(self, amount):
        if amount > Employee.MAX_BONUS:
            raise BonusError("The bonus amount is too high!")
        if self.salary + amount < Employee.MIN_SALARY:
            raise SalaryError("The salary after bonus is too low!")

        self.salary += amount


emp = Employee("Katze Rik", salary=50000)
try:
    emp.give_bonus(7000)
except SalaryError:
    print("SalaryError caught!")

try:
    emp.give_bonus(7000)
except BonusError:
    print("BonusError caught!")

try:
    emp.give_bonus(-100000)
except SalaryError:
    print("SalaryError caught again!")

try:
    emp.give_bonus(-100000)
except BonusError:
    print("BonusError caught again!")

# %%


# %%

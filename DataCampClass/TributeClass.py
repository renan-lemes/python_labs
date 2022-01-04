

# Define LoggedDF inherited from pd.DataFrame and add the constructor


class LoggedDF(pd.DataFrame):
    """
        Classe que pega e faz um cabeçalho no dataFrame
        no caso ela aplica uma subclasse no pandas e com isso
        utiliza-a para fazer a data que foi feita o DataFrame
    """

    def __init__(self, *args, **kwargs):
        pd.DataFrame.__init__(self, *args, **kwargs)
        self.created_at = datetime.today()


ldf = LoggedDF({"col1": [1, 2], "col2": [3, 4]})
print(ldf.values)
print(ldf.created_at)

# %%


# Define LoggedDF inherited from pd.DataFrame and add the constructor


class LoggedDF(pd.DataFrame):

    def __init__(self, *args, **kwargs):
        pd.DataFrame.__init__(self, *args, **kwargs)
        self.created_at = datetime.today()

    def to_csv(self, *args, **kwargs):
        # Copy self to a temporary DataFrame
        temp = self.copy()

        # Create a new column filled with self.created_at
        temp["created_at"] = self.created_at

        # Call pd.DataFrame.to_csv on temp, passing in *args and **kwargs
        pd.DataFrame.to_csv(temp, *args, **kwargs)

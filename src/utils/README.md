## Module: decorators

This module contains a decorator function called `injectable` that can be used to create injectable functions. The decorator wraps a function and returns a new function that allows partial application of the original function's arguments.

### Function: injectable(func)
This function is a decorator that takes a function `func` as input and returns a new function. The new function accepts any number of positional and keyword arguments (`*args` and `**kwargs`), which are passed to the `partial` function along with the original function `func` and its arguments. The `partial` function creates a new function that behaves like `func` with some of its arguments pre-filled.

Example usage:
```python
from utils.decorators import injectable

@injectable
def add(a, b):
    return a + b

add_5 = add(5)  # create a new function that adds 5 to any argument

result = add_5(3)  # result = 5 + 3 = 8
```

### Function: debounced(func)

The debounced decorator allows a function to be executed only if a specified amount of time has passed since the last execution with the same arguments. It takes an optional argument sec that represents the minimum time (in seconds) that needs to pass before executing the function again. By default, it is set to 3 seconds.

To use the debounced decorator, apply it to a function definition, specifying the desired time interval in seconds (optional). The decorated function will then exhibit the debouncing behavior, ensuring that it is executed only if the specified time has elapsed since the last execution with the same arguments.

Example usage:
```python
@debounced(sec=2)
def function1():
    print("Function 1 executed")

@debounced(sec=4)
def function2():
    print("Function 2 executed")

# Call function1 multiple times within 2 seconds - skipped
function1()
function1()
function1()

# Call function2 multiple times within 4 seconds - skipped
function2()
function2()
function2()

# Wait for 4 seconds
time.sleep(4)

# Call function1 and function2 after the debouncing intervals - both execute
function1()
function2()

# Output:
# -> Function 1 executed
# -> Function 2 executed
```
---

## Module: utils

This module contains various utility functions related to reading tables from a database, executing queries, and data typing.

### Function: read_tables(engine)
This function reads all tables from the specified database engine and returns a dictionary where the keys are table names and the values are pandas DataFrames containing the table data. It uses SQLAlchemy and pandas to perform the database query and data retrieval.

Parameters:
- `engine`: The SQLAlchemy database engine.

Returns:
- `tables`: A dictionary mapping table names to pandas DataFrames.

Example usage:
```python
from utils import read_tables
from sqlalchemy import create_engine

engine = create_engine("<Connection String>")
tables = read_tables(engine)
```

### Function: get_metadata(engine)
This function retrieves the metadata of the specified database engine using SQLAlchemy and returns it.

Parameters:
- `engine`: The SQLAlchemy database engine.

Returns:
- `metadata`: The metadata object representing the database schema.

Example usage:
```python
from utils import get_metadata
from sqlalchemy import create_engine

engine = create_engine("<Connection String>")
metadata = get_metadata(engine)
```

### Function: table_reader(engine, table_name, **kwargs)
This function reads a specific table from the database using the provided engine and table name. It also accepts additional keyword arguments that are passed to the `exec_query` function. It uses the `injectable` decorator to allow partial application of the `exec_query` function.

Parameters:
- `engine`: The SQLAlchemy database engine.
- `table_name`: The name of the table to read.
- `**kwargs`: Additional keyword arguments to pass to the `exec_query` function.

Returns:
- `table`: The SQLAlchemy Table object representing the table schema.
- `df`: A pandas DataFrame containing the table data.

Example usage:
```python
from utils import table_reader
from sqlalchemy import create_engine

engine = create_engine("<Connection String>")
read_table = table_reader(engine)

green_index = read_table("green_index", v=True, desc="Reading 'green_index'...")
livability_index = read_table("livability_index", v=True, desc="Reading 'livability_index'...")

read_table("private_db", q="UPDATE table SET ...", v=False, desc="Reading 'livability_index'...")
```

### Function: check_connection(engine, desc="")
This function checks the connection to the database engine by executing a test query (`SELECT 1`). If the query returns a scalar value of 1, it prints a success message. Otherwise, it prints a failure message and exits the program with an error code.

Parameters:
- `engine`: The SQLAlchemy database engine.
- `desc`: An optional description to include in the success message.

Example usage:
```python
from utils import check_connection
from sqlalchemy import create_engine

engine = create_engine("<Connection String>")
check_connection(engine, desc="We just landed on the moon.")
```

### Function: exec_query(engine, q: str, v=True, desc="")
This function executes a database query using the provided engine and query string. It also checks the connection to the database by calling the `check_connection` function unless the `v` (verbose) parameter is set to `False`. It returns the result of the query as a list of records.

Parameters:
- `engine`: The SQLAlchemy database engine.
- `q`: The query string to execute.
- `v`: A boolean flag indicating whether to perform connection checking (default is `True`).
- `desc`: An optional description to include in the connection check success message.

Returns:
- `res`: A list of records resulting from the query.

Example usage:
```python
from utils import exec_query
from sqlalchemy import create_engine

engine = create_engine("<Connection String>")
x_q = exec_query(engine)
result = x_q(q="Some amazing sql query...")
```

### Function: data_typing(df, integers=[], floats=[], strings=[], categoricals=[], dates={}, only_years=[], only_months=[], only_days=[])

This function performs data typing on a pandas DataFrame based on the specified column types. It supports typing columns as integers, floats, categoricals, and dates with optional format specifications. Additionally, it provides options to extract only the year, month, or day from a date column.

Parameters:
- `df`: The pandas DataFrame to perform data typing on.
- `integers`: A list of column names to be typed as integers.
- `floats`: A list of column names to be typed as floats.
- `strings`: A list of column names to be typed as strings.
- `categoricals`: A list of column names to be typed as categoricals.
- `dates`: A dictionary mapping date column names to their format strings (e.g., `{'date_column': '%Y-%m-%d'}`).
- `only_years`: A list of date column names to extract only the year.
- `only_months`: A list of date column names to extract only the month.
- `only_days`: A list of date column names to extract only the day.

Returns:
- `df`: The modified pandas DataFrame with the specified column types applied.

Note: The `injectable` decorator allows partial application of the `exec_query` function and `table_reader` function by creating new functions that can be used as standalone functions with some arguments pre-filled.

Examples demonstrating the usage of the `data_typing` function in each scenario:

1. Typing columns as integers:
```python
# Example DataFrame
df = pd.DataFrame({'A': ['1', '2', '3'], 'B': ['4', '5', '6']})

# Typing columns 'A' and 'B' as integers
df = data_typing(df, integers=['A', 'B'])

print(df.dtypes)
# Output:
# A    int8
# B    int8
# dtype: object
```

2. Typing columns as floats:
```python
# Example DataFrame
df = pd.DataFrame({'A': ['1.2', '2.3', '3.4'], 'B': ['4.5', '5.6', '6.7']})

# Typing columns 'A' and 'B' as floats
df = data_typing(df, floats=['A', 'B'])

print(df.dtypes)
# Output:
# A    float32
# B    float32
# dtype: object
```

3. Typing columns as categoricals:
```python
# Example DataFrame
df = pd.DataFrame({'A': ['apple', 'banana', 'orange'], 'B': ['red', 'green', 'blue']})

# Typing columns 'A' and 'B' as categoricals
df = data_typing(df, categoricals=['A', 'B'])

print(df.dtypes)
# Output:
# A    category
# B    category
# dtype: object
```

4. Typing columns as dates with format specification:
```python
# Example DataFrame
df = pd.DataFrame({'A': ['2021-01-01', '2021-02-01', '2021-03-01'], 'B': ['2021-04-01', '2021-05-01', '2021-06-01']})

# Typing column 'A' as date with format '%Y-%m-%d'
df = data_typing(df, dates={'A': '%Y-%m-%d'})

print(df.dtypes)
# Output:
# A    datetime64[ns]
# B            object
# dtype: object
```

5. Typing columns as dates and extracting only the year:
```python
# Example DataFrame
df = pd.DataFrame({'A': ['2021-01-01', '2022-02-01', '2023-03-01'], 'B': ['2024-04-01', '2025-05-01', '2026-06-01']})

# Typing column 'A' as date and extracting only the year
df = data_typing(df, dates={'A': '%Y-%m-%d'}, only_years=['A'])

print(df.dtypes)
# Output:
# A    period[A-DEC]
# B           object
# dtype: object
```

6. Typing columns as dates and extracting only the month:
```python
# Example DataFrame
df = pd.DataFrame({'A': ['2021-01-01', '2022-02-01', '2023-03-01'], 'B': ['2024-04-01', '2025-05-01', '2026-06-01']})

# Typing column 'A' as date and extracting only the month
df = data_typing(df, dates={'A': '%Y-%m-%d'}, only_months=['A'])

print(df.dtypes)
# Output:
# A    period[M]
# B      object
# dtype: object
```

7. Typing columns as dates and extracting only the day:
```python
# Example DataFrame
df = pd.DataFrame({'A': ['2021-01-01', '2022-02-01', '2023-03-01'], 'B': ['2024-04-01', '2025-05-01', '2026-06-01']})

# Typing column 'A' as date and extracting only the day
df = data_typing(df, dates={'A': '%Y-%m-%d'}, only_days=['A'])

print(df.dtypes)
# Output:
# A    period[D]
# B     object
# dtype: object
```

These examples illustrate the usage of the `data_typing` function in each scenario, demonstrating how it can be used to apply specific column types to a DataFrame based on the provided arguments.

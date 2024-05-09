from functools import partial, wraps
import inspect
import time


def injectable(func):
    """
    Dependency injection: initialize functions with a certain value that will be re-passed on every call.
    """
    return lambda *args, **kwargs: partial(func, *args, **kwargs)


def debounced(sec=3):
    """
    Decorator: Implements a debounce behavior for a function.

    The debounced decorator delays the execution of the wrapped function by the specified number of seconds (`sec`).
    If the wrapped function is called multiple times within the debounce period, only the last call will be executed
    after the debounce period has elapsed.

    Parameters:
        sec (float, optional): The debounce period in seconds (default is 3).

    Returns:
        function: The decorated function.

    Examples:
        ```
        @debounced(sec=2)
        def my_function():
            print("Executing my_function")

        # Only the last call within 2 seconds will be executed
        my_function()  # No output
        my_function()  # No output
        time.sleep(2)
        my_function()  # Output: "Executing my_function"
        ```
    """
    def decorate(f):
        pocket = {}

        @wraps(f)
        def end_func(*args, **kwargs):
            nonlocal pocket
            pocket_args = inspect.signature(f).bind(*args, **kwargs)
            pocket_args.apply_defaults()

            pocket_key = str(pocket_args)

            now = time.time()
            then = pocket.get(pocket_key, 0)

            def calc_time(n, t): return n - t

            cal_time = calc_time(now, then)

            if cal_time >= sec:
                res_returned = f(*args, **kwargs)
                pocket = {
                    k:
                    reg_time for k, reg_time in pocket.items()
                    if calc_time(now, reg_time) > sec
                }
                pocket[pocket_key] = now

                return res_returned
        return end_func
    return decorate

__all__ = ['injectable', 'debounced']

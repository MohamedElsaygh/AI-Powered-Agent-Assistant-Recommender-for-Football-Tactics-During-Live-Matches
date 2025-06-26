import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        out = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f}s")
        return out
    return wrapper

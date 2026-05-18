"""
Copyright: qiuzhi.tech
Author: hanyang
Date: 2025-08-28 14:54:22
LastEditTime: 2025-08-28 17:38:38
"""
import time


def print_green(x):
    return print(f"\033[92m {x}\033[00m")


def print_yellow(x):
    return print(f"\033[93m {x}\033[00m")


def print_red(x):
    return print(f"\033[91m {x}\033[00m")


def timed(fn):
    def wrapper(self, *args, **kwargs):
        start = time.time()
        ret = fn(self, *args, **kwargs)
        class_name = self.__class__.__name__ if hasattr(self, "__class__") else ""
        print_yellow(
            f"{class_name}.{fn.__name__} takes {(time.time() - start)*1000:.3f} ms"
        )
        return ret

    return wrapper

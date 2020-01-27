
import time

def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer

@parametrized
def timer(func, note):
	def wrapper(*args, **kwargs):
		t1 = time.time()
		result = func(*args, **kwargs)
		t2 = time.time()
		print("Elapsed time({}): {:.3f}".format(note, t2 - t1))
		return result
	return wrapper

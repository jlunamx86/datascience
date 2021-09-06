import time

def fibo_gen(max_iterations:int):
    n1 = 0
    n2 = 1
    counter = 0

    while counter <= max_iterations:
        if counter == 0:
            counter += 1
            yield n1
        elif counter == 1:
            counter += 1
            yield n2
        else:
            aux = n1 + n2
            n1, n2 = n2, aux
            counter += 1
            yield aux

if __name__ == "__main__":
    max_iterations = int(input("Max number of iterations:"))

    fibonacci = fibo_gen(max_iterations)
    for elem in fibonacci:
        print(elem)
        time.sleep(1)
import time


class FiboIter():

    def __init__(self, iterations):
        self.iterations = iterations

    def __iter__(self):
        self.n1 = 0
        self.n2 = 1
        self.counter = 0
        return self        

    def __next__(self):
        if not self.iterations or self.counter <= self.iterations:
            if self.counter == 0:
                self.counter += 1
                return self.n1
            elif self.counter == 1:
                self.counter += 1
                return self.n2
            else:
                self.aux = self.n1 + self.n2
                #swapping
                self.n1, self.n2 = self.n2, self.aux
                self.counter += 1
                return self.aux
        else:
            raise StopIteration


if __name__ == "__main__":
    max_iterations = int(input("Max number of iterations:"))

    fibonacci = FiboIter(max_iterations)
    for elem in fibonacci:
        print(elem)
        #slow down to view result
        time.sleep(1)
from loaddata import get_MNIST


if __name__ == "__main__":
    (x, xl), (y, yl) = get_MNIST()
    print(x.shape)
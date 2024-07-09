# Assignment 01

# Name: Sahil Pramod Raut
# Student Id: CT_S_PCCOER_DATA_SCIENCE_213
# Contact No: 9175663960
# Email ID: raut.sahil003@gmail.com

def lower_triangular(n):
    for i in range(1, n + 1):
        print('*' * i)
    print()  # Add a newline for separation


def upper_triangular(n):
    for i in range(n, 0, -1):
        print('*' * i)
    print()  # Add a newline for separation


def pyramid(n):
    for i in range(1, n + 1):
        print(' ' * (n - i) + '*' * (2 * i - 1))
    print()  # Add a newline for separation


if __name__ == "__main__":
    size = 5  # You can change this value to generate different sizes

    print("Lower Triangular Pattern:")
    lower_triangular(size)

    print("Upper Triangular Pattern:")
    upper_triangular(size)

    print("Pyramid Pattern:")
    pyramid(size)

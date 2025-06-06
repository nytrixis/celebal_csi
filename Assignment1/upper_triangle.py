def upper_triangle(rows):
    for i in range(rows):
        for j in range(i):
            print('  ', end='')
        for k in range(rows-i):
            print('*', end=' ')
        print()

n=int(input("Enter no of rows: "))
upper_triangle(n)
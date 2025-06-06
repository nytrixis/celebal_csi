def lower_triangle(rows):
    for i in range(1, rows+1):
        for j in range(i):
            print('*', end=' ')
        print()

n=int(input("Enter no of rows: "))
lower_triangle(n)
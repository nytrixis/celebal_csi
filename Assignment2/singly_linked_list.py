class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def add_end(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = new_node

    def print_list(self):
        curr = self.head
        if not curr:
            print("List is empty.")
            return
        while curr:
            print(curr.data, end=" -> " if curr.next else "\n")
            curr = curr.next

    def delete_nth(self, n):
        if not self.head:
            raise Exception("Cannot delete from an empty list.")
        if n < 1:
            raise Exception("Index must be 1 or greater.")
        if n == 1:
            self.head = self.head.next
            return
        curr = self.head
        prev = None
        count = 1
        while curr and count < n:
            prev = curr
            curr = curr.next
            count += 1
        if not curr:
            raise Exception("Index out of range.")
        prev.next = curr.next

if __name__ == "__main__":
    ll = LinkedList()
    try:
        n = int(input("Enter the size of the linked list: "))
        for i in range(n):
            val = int(input(f"Enter value {i+1}: "))
            ll.add_end(val)
        print("Initial list:")
        ll.print_list()

        while True:
            print("\nOptions:\n1. Add at end\n2. Delete nth node\n3. Print list\n4. Exit")
            choice = input("Enter your choice: ")
            if choice == '1':
                val = int(input("Enter value to add at end: "))
                ll.add_end(val)
                print("Updated list:")
                ll.print_list()
            elif choice == '2':
                idx = int(input("Enter position to delete (1-based index): "))
                try:
                    ll.delete_nth(idx)
                    print("Updated list:")
                    ll.print_list()
                except Exception as e:
                    print("Error:", e)
            elif choice == '3':
                ll.print_list()
            elif choice == '4':
                print("Exiting.")
                break
            else:
                print("Invalid choice. Try again.")
    except Exception as e:
        print("Error:", e)
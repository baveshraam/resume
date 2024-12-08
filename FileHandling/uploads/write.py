import os

# Open and write to the file
f = open("newfile.txt", "w")
f.write("woops i just deleted the file")
f.close()  # Close the file after writing

# Reopen the file in read mode to read its contents
f = open("newfile.txt", "r")  # Open the file in read mode
print(f.read())  # Read and print the file content
f.close()  # Close the file after reading

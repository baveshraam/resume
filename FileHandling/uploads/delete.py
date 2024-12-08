import os
if os.path.exists("deletethis.txt"):
    os.remove("deletethis.txt")
else:
    f = open("deletethis.txt","x")
    f.close()
    os.remove("deletethis.txt")
import matplotlib.pyplot as plt
import numpy as np
import os

username = input("Enter username:")
print(len(username))

if len(username) > 16:
    print( username + " will  Become what he wants to be")
elif len(username) == 16:
    print(username + " IPS")

else:
    print(username + " IAS")
print(len(username))


f = open("Hakkim1.txt","w")
f.write("Never Give up")
f.close()

# f = open("Hakkim1.txt","a")
# f.write("I am  Great, I do best")
# f.close()

# f = open("Hakkim1.txt","r")
# print(f.read())

os.remove("Hakkim1.txt")
           
# del username
# try:
#    f = open("Hakkim_Todo list.txt",'r')
#    f.write("I am really Doing Great")
#    print(f.read())
# except:
#     print("User name not valid")
# else:
#     print("Do well Bro")    
# finally:
#     print(username + ' Doing Great')    
    
    

        

# x = np.linspace(10, 3, 10)  # Create a list of evenly-spaced numbers over the range
# plt.plot(x, np.sin(x))       # Plot the sine of each x point
# plt.show()  

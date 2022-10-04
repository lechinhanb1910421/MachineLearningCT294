from filecmp import cmp
import pandas as pd
import numpy as np


print("Simple if condition")
a = 5
b = 3
if a > b:
    a = a*2+3
    b -= 6
    c = a/b
    print(c)

print("\nSeparate a line to miltiple lines by \"\\\" symbol")
c = a+b +\
    10*a - b/4 - \
    5+a*3
print(c)

print("\nOr simply surround them with ( )")
c2 = (a+b +
      10*a - b/4 -
      5+a*3)
print(c2)


print("\nIf else condition")
# a = 5
# b = 3
if a > b:
    print("True")
    print(a)
else:
    print("false")
    print(b)

print("\nWhile condition")
a, b = 1, 10
while a < b:
    a += 1
    print(a)

print("\nFor loop in range")
for i in range(1, 10):
    print(i, end=" ")

print("\n\nDefine a function")


def square(number):
    return number*number


print(square(7))

print("\nData types in Python")
a = 5
b = -7
c = 1.234

str1 = "Hello"
str2 = 'welcome'
str3 = "abcdef12345"

print(a, end=" ")
print(type(a))
print(b, end=" ")
print(type(b))
print(c, end=" ")
print(type(c))

print(str1, end=" ")
print(type(str1))
print(str2, end=" ")
print(type(str2))
print(str3, end=" ")
print(type(str3))

print("\n\nList in Python is dynamically allcated memory, which means we can modify the list after initialize it")
cats = ['Tom', 'Snappy', 'Kitty', 'Jessie', 'Chester']
print(cats[2])
print(cats[-1])
print(cats[1:3])

print("Modify the list")
print(cats)
cats.append('Jerry')
print(cats)

cats[-1] = 'Jerry Cat'
print(cats)

del cats[1]
print(cats)


print("Tupple in Python is immutable which means we can not modify it after initialize")
catsTupple = ('Tom', 'Snappy', 'Kitty', 'Jessie', 'Chester')
print(len(catsTupple))
print(catsTupple)

print("\nCreate a tuple from list")
tupleFromList = tuple(cats)
print(tupleFromList)

print("\nDictionary in Python")
myDict = {'Name': 'Everett', 'Age': 21, 'Class': 'DI19V7A8'}
myDict2 = {'Name': 'Nhan Le Nguyen Chi', 'Age': 21, 'Class': 'DI19V7A8'}
print(myDict['Name'])
print(myDict['Age'])

print("\nAppend a dictionary")
myDict['School'] = "Can Tho University"
print(myDict)

print("\n\"has_key\" method was removed in Python 3. We can use \"in\" keyword instead")
# if myDict.has_key('Address'):
#     print(myDict['Address'])
# else:
#     print("There is no key named Address")
if 'Name' in myDict:
    print(myDict['Name'])
else:
    print("There is no key named Address")

print("\n\nWorking with Numpy Library!!")

a = np.array([0, 1, 2, 3, 4, 5])

print(a)

print("\na has " + str(a.ndim) + " dimension(s)")

print("Shape of a: " + str(a.shape))

print(a[a > 3])
a[a > 3] = 10
print(a)

print("\nReshape a to ")
b = a.reshape((3, 2))
print(b)

print(b[2][1])

b[2][0] = 50

c = b*2
print(c)

print("\n\nWorking with Pandas library!!")
# play_tennis.csv file be placed in the same folder with .py file for easy file_path parameter
data = pd.read_csv("play_tennis.csv", delimiter=',')
print("\nFirst 5 rows")
print(data.head())
print("\nLast 7 rows")
print(data.tail(7))

print("\nRow 3th to row 8th")
print(data.loc[3:8])

print("\nCol 3th to col 5th")
print(data.iloc[:, 2:6])

print("\nRow 5th to 9th of 3th col")
print(data.iloc[5:10, 3:4])

print("\nColumn named \'Outlook\'")
print(data.Outlook)

# print("\nValues of column named \'Outlook\'")
# OutlookValuesList = data.Outlook.tolist()
# print(OutlookValuesList)

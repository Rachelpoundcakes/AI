mylist = ['Korean', 'English', 'Math', 'Science', 'Politics']
for n, name in enumerate(mylist): # enumerate: 순서와 리스트의 값을 반환하는 기능
    print("Course: {}, Number: {}".format(name, n))
"""
Course: Korean, Number: 0
Course: English, Number: 1 
Course: Math, Number: 2    
Course: Science, Number: 3 
Course: Politics, Number: 4
"""
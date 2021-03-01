from sklearn import tree
my_tree=tree.ExtraTreeClassifier()
tree.DecisionTreeClassifier
dactrung=[
        [1,2,2,0],
        [0,1,0,1],
        [1,1,0,1],
        [0,0,0,2],
        [1,0,0,0],
        [2,1,2,0],
        [2,2,2,1],
        [0,1,1,0]]
kq=[0,1,1,0,0,0,0,1]
rs=my_tree.fit(dactrung,kq)
temp=rs.predict([[1,1,0,1]])
print(temp)

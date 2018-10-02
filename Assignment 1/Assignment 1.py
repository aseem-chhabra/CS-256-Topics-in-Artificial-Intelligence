import pandas as pd

def activity():

    x_data = [[0,0,0],[0,1,0],[1,0,1],
              [1,1,0],[1,0,0],[1,1,1],
              [0,1,1],[0,1,1]
         ]
    x = pd.DataFrame(x_data, columns=['X1','X2','X3'])
    print(x)

    y_data = [[1],[1],[0],[1],[1],[0],[0],[0]]
    y = pd.DataFrame(y_data,columns = ['Y'])
    print(y)

    print(list(x.columns.values))
    print(list(y.columns.values))

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

    print(X_train)
    print("\n")
    print(y_train)

    from sklearn.tree import DecisionTreeClassifier
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    print("\n")
    print(dtree.predict(X_test))

    #features = x.columns
    from sklearn.externals.six import StringIO
    from sklearn.tree import export_graphviz
    import pydotplus
    dot_data = StringIO()
    export_graphviz(dtree,
                    out_file=dot_data,
                    feature_names=x.columns,
                    class_names=['0', '1'],
                    filled=True, rounded=True,
                    impurity=False)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("tree.png")

def main():
    activity()

if __name__ == '__main__':
    main()
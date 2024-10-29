from sqlalchemy import create_engine,MetaData, Table, Column, Integer, String
from sqlalchemy.orm import sessionmaker 
from pathlib import Path
from bokeh.plotting  import figure,show,ColumnDataSource
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
def main():
    # path= Path('./data/test.db')
    # print(path)
    # engine= create_engine(f'sqlite:///{path}',echo=True)
    # Session = sessionmaker(engine)
    # session=Session()
    # meta = MetaData()

    train_data=None
    test_data=None
    ideal_data=None
    csv_files=['ideal.csv','test.csv','train.csv']

    for file in csv_files:
        path= Path(f'./assets/{file}')
        if file=="ideal.csv":
            ideal_data= pd.read_csv(path)
        elif file=="test.csv":
            test_data= pd.read_csv(path)
        else:
            train_data=pd.read_csv(path)

    print("Number of samples in train data:", len(train_data))
    print("Number of samples in test data:", len(test_data))
    print("Number of samples in ideal data:", len(ideal_data))

    X_train = train_data['x'] 
    y_train = train_data.drop('x', axis=1)

    model = LinearRegression()
    model.fit(X_train,y_train)
    ideal_functions = ideal_data.values  # Assuming each row in 'ideal.csv' represents a function
    # print(ideal_functions)
    rmse_scores = []




    # mse_ml_model = mean_squared_error(x_train, model.predict(Y_train))
    # print("Mean Squared Error (Multiple Linear Regression Model):", mse_ml_model)

    # print(x_train)
    # for function in ideal_functions:
    #     # mean_squared_error(x_train, function)
    #     print(len(function))
    #     print(function)
        # rmse_scores.append()


    # top_four_indices = np.argsort(rmse_scores)[:4]
    # top_four_functions = ideal_functions[top_four_indices]

    # p = figure(title="Top Four Ideal Functions", x_axis_label='Index', y_axis_label='Value')

    # for i, function in enumerate(top_four_functions):
    #     p.line(range(len(function)), function, legend_label=f'Function {i+1}', line_color=f'color{i+1}')

    # show(p)
    # df=pd.read_csv(path)
    # print(df)


    # print(test_data)
    # sqlite_path= Path('./data/test.db')
    # print(sqlite_path)

    # source=ColumnDataSource(df)


    # print(source)
    # engine= create_engine(f'sqlite:///{sqlite_path}',echo=True)
    # # Session = sessionmaker(engine)
    # df.to_sql("ideal", engine, index=False, if_exists='replace')

    # Y = df.drop(columns=['x'])
    # x = df['x']


    # p = figure(title="Multiple linear regression")
    # for col in Y.columns:
    #     p.circle(x,df[col],size=8)

    # show(p)


    # p.line(x, y, legend_label="Temp.", line_width=2)
    # show(p)
    # students = Table(
    #     'students', meta, 
    #     Column('id', Integer, primary_key = True), 
    #     Column('name', String), 
    #     Column('lastname', String),
    # )
    # meta.create_all(engine)

if __name__ == "__main__":
    main()
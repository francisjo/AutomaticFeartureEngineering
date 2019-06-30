import pandas as pd


def load_data_online():
    titanic = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/train.csv'
    car = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/car.csv'
    adult = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/adult.csv'
    heart = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/heart.csv'
    bridges = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/bridges.csv'
    audiology = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/audiologystandardized.csv'
    car1 = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/car1.csv'
    random = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/random.csv'
    new_dataset = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/new_dataset.csv'
    nursery = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/nursery.csv'
    books = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/good_reads_final.csv'

    df_titanic = pd.read_csv(titanic)
    df_car = pd.read_csv(car)
    df_adult = pd.read_csv(adult)
    df_heart = pd.read_csv(heart)
    df_bridges = pd.read_csv(bridges)
    df_audiology = pd.read_csv(audiology)
    df_car1 = pd.read_csv(car1)
    df_random = pd.read_csv(random)
    df_new_dataset = pd.read_csv(new_dataset)
    df_nursery = pd.read_csv(nursery)
    df_books = pd.read_csv(books, encoding="ISO-8859-1")
    df_dict = {"Titanic": df_titanic,
               "Car1": df_car1,
               "Car": df_car,
               "Adult": df_adult,
               "Audiology": df_audiology,
               "Bridges": df_bridges,
               "Random": df_random,
               "Heart": df_heart,
               "Nursery": df_nursery,
               "Books": df_books,
               "New_dataset": df_new_dataset
               }
    return df_dict
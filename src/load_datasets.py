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

    df_titanic = pd.read_csv(titanic)
    df_car = pd.read_csv(car)
    df_adult = pd.read_csv(adult)
    df_heart = pd.read_csv(heart)
    df_bridges = pd.read_csv(bridges)
    df_audiology = pd.read_csv(audiology)
    df_car1 = pd.read_csv(car1)
    df_random = pd.read_csv(random)
    df_new_dataset = pd.read_csv(new_dataset)
    df_dict = {"titanic": df_titanic
               #"car1": df_car1,
               #"car": df_car,
               #"adult": df_adult,
               #"audiology": df_audiology,
               #"bridges": df_bridges,
               #"random": df_random,
               #"heart": df_heart,
                } #, "new_dataset": df_new_dataset
    return df_dict


def load_data_local():

    titanic = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\train.csv'
    car = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\car.csv'
    adult = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\adult.csv'
    heart = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\heart.csv'
    bridges = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\bridges.csv'
    audiology = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\audiologystandardized.csv'
    car1 = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\car1.csv'
    random = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\random.csv'
    new_dataset = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\new_dataset.csv'
    df_titanic = pd.read_csv(titanic)
    df_car = pd.read_csv(car)
    df_adult = pd.read_csv(adult)
    df_heart = pd.read_csv(heart)
    df_bridges = pd.read_csv(bridges)
    df_audiology = pd.read_csv(audiology)
    df_car1 = pd.read_csv(car1)
    df_random = pd.read_csv(random)
    df_new_dataset = pd.read_csv(new_dataset)
    df_dict = {"titanic": df_titanic,
               "car1": df_car1,
               "car": df_car,
               "adult": df_adult,
               "audiology": df_audiology,
               "bridges": df_bridges,
               "random": df_random,
               "heart": df_heart,
               }  # , "new_dataset": df_new_dataset
    return df_dict
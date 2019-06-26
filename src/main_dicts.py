import category_encoders as ce
import label_encoder as le
import frequency_encoder as fe

def get_groundtruth_dict():
    groundtruth_dict = {
        "adult":
            {
                "age": "Numerical",
                "workclass": "Nominal",
                "fnlwgt": "Numerical",
                "education": "Ordinal",
                "education-num": "Ordinal",
                "marital-status": "Nominal",
                "occupation": "Nominal",
                "relationship": "Nominal",
                "race": "Nominal",
                "sex": "Nominal",
                "capital-gain": "Numerical",
                "capital-loss": "Numerical",
                "hours-per-week": "Numerical",
                "native-country": "Nominal",
                "class": "Nominal",
            },
        "car":
            {
                "make": "Nominal",
                "fuel_type": "Nominal",
                "aspiration": "Nominal",
                "body_style": "Nominal",
                "drive_wheels": "Nominal",
                "engine_location": "Nominal",
                "wheel_base": "Numerical",
                "length": "Numerical",
                "width": "Numerical",
                "height": "Numerical",
                "engine_type": "Nominal",
                "num_of_cylinders": "Nominal",
                "engine_size": "Numerical",
                "fuel_system": "Nominal",
                "compression_ratio": "Numerical",
                "horsepower": "Numerical",
                "peak_rpm": "Numerical",
                "city_mpg": "Numerical",
                "highway_mpg": "Numerical",
                "price": "Numerical",
                "curb_weight": "Numerical",
                "num_of_doors_num": "Numerical",
                "num_of_cylinders_num": "Numerical"
            },
        "titanic":
            {
                "PassengerId": "Nominal",
                "Survived": "Nominal",
                "Pclass": "Ordinal",
                "Name": "Nominal",
                "Sex": "Nominal",
                "Age": "Numerical",
                "SibSp": "Numerical",
                "Parch": "Numerical",
                "Ticket": "Nominal",
                "Fare": "Numerical",
                "Cabin": "Nominal",
                "Embarked": "Nominal",
            },
        "bridges":
            {
                "IDENTIF": "Nominal",
                "RIVER": "Nominal",
                "LOCATION": "Numerical",
                "ERECTED": "Numerical",
                "PURPOSE": "Nominal",
                "LENGTH": "Numerical",
                "LANES": "Numerical",
                "CLEAR-G": "Nominal",
                "T-OR-D": "Nominal",
                "MATERIAL": "Nominal",
                "SPAN": "Ordinal",
                "REL-L": "Nominal",
                "binaryClass": "Nominal"
            },
        "heart":
            {
                "age": "Numerical",
                "sex": "Nominal",
                "cp": "Nominal",
                "trestbps": "Numerical",
                "chol": "Numerical",
                "fbs": "Nominal",
                "restecg": "Nominal",
                "thalach": "Numerical",
                "exang": "Nominal",
                "oldpeak": "Numerical",
                "slope": "Ordinal",
                "ca": "Nominal",
                "thal": "Ordinal",
                "target": "Nominal",
            },
        "audiology":
            {
                "air": "Ordinal",
                "ar_c": "Nominal",
                "ar_u": "Nominal",
                "o_ar_c": "Nominal",
                "o_ar_u": "Nominal",
                "speech": "Ordinal",
                "indentifier": "Nominal",
                "class": "Nominal"
            },
        "car1":
            {
                "buying": "Nominal",
                "maint": "Ordinal",
                "doors": "Numerical",
                "persons": "Numerical",
                "lug_boot": "Ordinal",
                "safety": "Ordinal"
            },
        "random":
            {
                "Color": "Nominal",
                "Size": "Ordinal",
                "Act": "Nominal",
                "Age": "Nominal",
                "Inflated": "Nominal"
            }
    }
    return groundtruth_dict


def get_target_variables_dicts():
    target_dict = {"adult": "class",
                   "car": "price",
                   "titanic": "Survived",
                   "bridges": "binaryClass",
                   "heart": "target",
                   "audiology": "class",
                   "car1": "safety",
                   "random": "Inflated"}
    return target_dict


def get_encoder_dict():
    encoder_dict = {'OneHotEncoder': ce.OneHotEncoder(),
                    'BinaryEncoder': ce.BinaryEncoder(),
                    'HashingEncoder': ce.HashingEncoder(),
                    'LabelEncoder': le.MultiColumnLabelEncoder(),
                    'FrequencyEncoder': fe.FrequencyEncoder(),
                    'TargetEncoder': ce.TargetEncoder(),
                    'HelmertEncoder': ce.HelmertEncoder(),
                    'JamesSteinEncoder': ce.JamesSteinEncoder(),
                    'BaseNEncoder': ce.BaseNEncoder(),
                    'SumEncoder': ce.SumEncoder(),
                    }
    return encoder_dict
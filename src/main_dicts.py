import category_encoders as ce
import label_encoder as le
import frequency_encoder as fe


def get_groundtruth_dict():
    groundtruth_dict = {
        "Adult":
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
        "Car":
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
        "Titanic":
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
        "Bridges":
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
        "Heart":
            {
                "age": "Numerical",
                "sex": "Nominal",
                "cp": "Numerical",
                "trestbps": "Numerical",
                "chol": "Numerical",
                "fbs": "Numerical",
                "restecg": "Numerical",
                "thalach": "Numerical",
                "exang": "Numerical",
                "oldpeak": "Numerical",
                "slope": "Numerical",
                "ca": "Numerical",
                "thal": "Numerical",
                "target": "Nominal",
            },
        "Audiology":
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
        "Car1":
            {
                "buying": "Nominal",
                "maint": "Ordinal",
                "doors": "Numerical",
                "persons": "Numerical",
                "lug_boot": "Ordinal",
                "safety": "Ordinal",
                "class": "Nominal"
            },
        "Random":
            {
                "Color": "Nominal",
                "Size": "Ordinal",
                "Act": "Nominal",
                "Age": "Nominal",
                "Inflated": "Nominal"
            },
        "New_dataset":
            {
                "size": "Ordinal",
                "temp": "Ordinal",
                "level": "Ordinal",
                "color": "Nominal",
            },
        "Nursery":
            {
                "parents": "Nominal",
                "has_nurs": "Ordinal",
                "form": "Nominal",
                "children": "Numerical",
                "housing": "Ordinal",
                "finance": "Nominal",
                "social": "Ordinal",
                "health": "Nominal"
            },
        "Books":
            {
                "author_average_rating": "Numerical",
                "author_gender": "Nominal",
                "author_genres": "Nominal",
                "author_id": "Nominal",
                "author_name": "Nominal",
                "author_page_url": "Nominal",
                "author_rating_count": "Numerical",
                "author_review_count": "Numerical",
                "birthplace": "Nominal",
                "book_average_rating": "Numerical",
                "book_fullurl": "Nominal",
                "book_title": "Nominal",
                "genre_1": "Nominal",
                "genre_2": "Nominal",
                "num_ratings": "Numerical",
                "num_reviews": "Numerical",
                "pages": "Numerical",
                "new score": "Nominal"
            }
    }
    return groundtruth_dict


def get_target_variables_dicts():
    target_dict = {"Adult": "class",
                   "Car": "price",
                   "Titanic": "Survived",
                   "Bridges": "binaryClass",
                   "Heart": "target",
                   "Audiology": "class",
                   "Car1": "class",
                   "Random": "Inflated",
                   "Nursery": "health",
                   "books": "new score"}
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
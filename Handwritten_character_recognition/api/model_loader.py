import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras


def load_models():
    models = {}

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    search_paths = [
        project_root,
        os.path.join(project_root, "models"),
        os.path.join(project_root, "App"),
        current_dir,
    ]

    possible_filenames = [
        "emnist_model.keras",
        "emnist_main.h5",
        "character_model.h5",
    ]

    emnist_model_found = False
    for base_path in search_paths:
        for filename in possible_filenames:
            path = os.path.join(base_path, filename)
            if os.path.exists(path):

                try:
                    models['emnist'] = keras.models.load_model(path)
                    emnist_model_found = True
                except Exception as e:

                    models['emnist'] = None
                break
        if emnist_model_found:
            break

    if not emnist_model_found:

        models['emnist'] = None

    return models


def create_dummy_model():

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(62, activation='softmax')
    ])
    return model


if __name__ == '__main__':
    models = load_models()

    if models['emnist'] is None:
        print("Using dummy model as fallback.")
        models['emnist'] = create_dummy_model()

    models['emnist'].summary()

import os
from tensorflow.keras.models import load_model
import numpy as np
from flask import  jsonify, request


class SmartModelMerger:
    def __init__(self, search_directories=None):
        if search_directories is None:
            self.search_directories = [
                "models",
                ".",
                "api/models",
                "../models",
                "saved_models",
                "checkpoints"
            ]
        else:
            self.search_directories = search_directories

    def find_all_model_files(self):
        found_models = {}

        print(" Searching for model files...")

        for directory in self.search_directories:
            if os.path.exists(directory):
                print(f"Checking directory: {directory}")

                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if file.endswith(('.h5', '.keras')):
                            full_path = os.path.join(root, file)
                            file_size = os.path.getsize(full_path) / (1024 * 1024)
                            found_models[file] = {
                                'path': full_path,
                                'directory': root,
                                'size_mb': round(file_size, 2)
                            }
                            print(f"  Found: {file} ({file_size:.2f} MB) in {root}")
            else:
                print(f"  Directory not found: {directory}")

        return found_models

    def list_emnist_models(self):
        all_models = self.find_all_model_files()
        emnist_models = {k: v for k, v in all_models.items()
                         if 'emnist' in k.lower()}

        print(f"\n Found {len(emnist_models)} EMNIST models:")
        for name, info in emnist_models.items():
            print(f"  • {name} -> {info['path']} ({info['size_mb']} MB)")

        return emnist_models

    def auto_detect_models_to_merge(self):
        emnist_models = self.list_emnist_models()

        if len(emnist_models) < 2:
            print(f"Need at least 2 EMNIST models to merge. Found: {len(emnist_models)}")
            return None

        sorted_models = sorted(emnist_models.items(),
                               key=lambda x: x[1]['size_mb'],
                               reverse=True)

        print(f"\n Auto-selecting top 2 models for merging:")
        model1_name, model1_info = sorted_models[0]
        model2_name, model2_info = sorted_models[1]

        print(f"  1. {model1_name} ({model1_info['size_mb']} MB)")
        print(f"  2. {model2_name} ({model2_info['size_mb']} MB)")

        return [model1_info['path'], model2_info['path']]

    def load_checkpoint_weights(self, model_path):
        try:
            print(f" Loading model: {model_path}")
            model = load_model(model_path)
            print(f"   Successfully loaded: {model_path}")
            return model.get_weights(), model
        except Exception as e:
            print(f"  Error loading {model_path}: {str(e)}")
            return None, None

    def check_model_compatibility(self, model1, model2):
        try:
            weights1 = model1.get_weights()
            weights2 = model2.get_weights()

            if len(weights1) != len(weights2):
                print(f" Models have different number of layers: {len(weights1)} vs {len(weights2)}")
                return False

            for i, (w1, w2) in enumerate(zip(weights1, weights2)):
                if w1.shape != w2.shape:
                    print(f"Layer {i} shape mismatch: {w1.shape} vs {w2.shape}")
                    return False

            print("Models are compatible for merging!")
            return True

        except Exception as e:
            print(f" Error checking compatibility: {str(e)}")
            return False

    def merge_weights(self, weights_list):
        if not weights_list or len(weights_list) < 2:
            raise ValueError("Need at least 2 models to merge")

        print(f"Averaging weights from {len(weights_list)} models...")

        merged_weights = [np.zeros_like(w) for w in weights_list[0]]

        for weights in weights_list:
            for i, layer_weights in enumerate(weights):
                merged_weights[i] += layer_weights

        for i in range(len(merged_weights)):
            merged_weights[i] /= len(weights_list)

        print("Weights merged successfully!")
        return merged_weights

    def smart_merge(self, output_path="models/emnist_model.h5", model_paths=None):

        if model_paths is None:
            model_paths = self.auto_detect_models_to_merge()
            if model_paths is None:
                return False

        print(f"\nStarting merge process...")
        print(f" Input models: {model_paths}")
        print(f" Output model: {output_path}")

        all_weights = []
        base_model = None
        successful_loads = []

        for path in model_paths:
            weights, model = self.load_checkpoint_weights(path)
            if weights is not None and model is not None:
                all_weights.append(weights)
                if base_model is None:
                    base_model = model
                successful_loads.append(path)
            else:
                print(f" Skipping {path} due to loading error")

        if len(all_weights) < 2:
            print(f" Could only load {len(all_weights)} models. Need at least 2 for merging.")
            return False

        print(f"Successfully loaded {len(all_weights)} models")

        if len(all_weights) >= 2:
            temp_model1 = load_model(successful_loads[0])
            temp_model2 = load_model(successful_loads[1])

            if not self.check_model_compatibility(temp_model1, temp_model2):
                print(" Models are not compatible for merging!")
                return False

        try:
            merged_weights = self.merge_weights(all_weights)

            base_model.set_weights(merged_weights)

            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f" Created directory: {output_dir}")

            base_model.save(output_path)

            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                print(f" SUCCESS! Merged model saved to: {output_path}")
                print(f" Merged model size: {file_size:.2f} MB")

                try:
                    test_model = load_model(output_path)
                    print(" Merged model verified - loads successfully!")
                    return True
                except Exception as e:
                    print(f"⚠  Warning: Merged model saved but failed verification: {str(e)}")
                    return False
            else:
                print(" Failed to save merged model")
                return False

        except Exception as e:
            print(f" Error during merging: {str(e)}")
            return False


def merge_available_models():

    merger = SmartModelMerger()

    print("=" * 60)
    print(" SMART MODEL MERGER")
    print("=" * 60)

    all_models = merger.find_all_model_files()

    if not all_models:
        print("No model files found!")
        print(" Make sure your .h5 or .keras files are in one of these directories:")
        for directory in merger.search_directories:
            print(f"   • {directory}")
        return False

    success = merger.smart_merge()

    if success:
        print("\n MODEL MERGING COMPLETED SUCCESSFULLY!")
        print(" You can now use 'models/emnist_model.h5' in your Flask app")
    else:
        print("\nMODEL MERGING FAILED!")
        print("Check the error messages above for details")

    return success


def add_smart_merge_routes(app):

    @app.route('/find_models', methods=['GET'])
    def find_models_route():

        merger = SmartModelMerger()
        models = merger.find_all_model_files()
        emnist_models = merger.list_emnist_models()

        return jsonify({
            "status": "success",
            "all_models": len(models),
            "emnist_models": len(emnist_models),
            "models": {
                "all": models,
                "emnist": emnist_models
            }
        })

    @app.route('/smart_merge', methods=['POST'])
    def smart_merge_route():

        try:
            merger = SmartModelMerger()

            data = request.get_json() or {}
            output_path = data.get('output_path', 'models/emnist_model.h5')
            model_paths = data.get('model_paths')  # Optional: specify exact paths

            success = merger.smart_merge(output_path, model_paths)

            if success:
                return jsonify({
                    "status": "success",
                    "message": "Models merged successfully!",
                    "merged_model": output_path
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Failed to merge models. Check logs for details."
                })

        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Error during merge: {str(e)}"
            })


if __name__ == "__main__":
    merge_available_models()
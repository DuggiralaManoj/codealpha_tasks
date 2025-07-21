import numpy as np

from PIL import Image
import base64
import io


EMNIST_CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'


def predict_character(image_data, model, debug=False):

    try:
        if debug:
            print("Starting image preprocessing...")

        pil_image = None


        if hasattr(image_data, 'read'):
            if debug:
                print("Processing file upload...")
            image_bytes = image_data.read()
            pil_image = Image.open(io.BytesIO(image_bytes))


        elif isinstance(image_data, str):
            if debug:
                print("Processing base64 string...")

            if image_data.startswith('data:image'):
                image_data = image_data.split(',', 1)[1]

            image_bytes = base64.b64decode(image_data)
            pil_image = Image.open(io.BytesIO(image_bytes))


        elif isinstance(image_data, Image.Image):
            pil_image = image_data

        else:
            return {
                "success": False,
                "error": "IMAGE_UNSUPPORTED_FORMAT",
                "message": f"Unsupported image format: {type(image_data)}",
                "prediction": None,
                "confidence": 0.0,
                "all_predictions": []
            }

        if debug:
            print(f"Original image size: {pil_image.size}")
            print(f"Original image mode: {pil_image.mode}")

        processed_image = preprocess_for_emnist(pil_image, debug=debug)

        if processed_image is None:
            return {
                "success": False,
                "error": "IMAGE_PREPROCESS_FAILED",
                "message": "Failed to preprocess image for EMNIST",
                "prediction": None,
                "confidence": 0.0,
                "all_predictions": []
            }


        if debug:
            print(" Making prediction...")

        predictions = model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])


        predicted_char = EMNIST_CHARACTERS[predicted_class] if predicted_class < len(EMNIST_CHARACTERS) else '?'


        top_indices = np.argsort(predictions[0])[::-1][:5]
        all_predictions = []

        for idx in top_indices:
            char = EMNIST_CHARACTERS[idx] if idx < len(EMNIST_CHARACTERS) else '?'
            conf = float(predictions[0][idx])
            all_predictions.append({
                "char": char,
                "confidence": conf
            })

        if debug:
            print(f"✅ Prediction successful: {predicted_char} ({confidence:.3f})")

        return {
            "success": True,
            "prediction": predicted_char,
            "confidence": confidence,
            "all_predictions": all_predictions,
            "error": None,
            "message": "Prediction successful"
        }

    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        if debug:
            print(f" {error_msg}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")

        return {
            "success": False,
            "error": "PREDICTION_FAILED",
            "message": error_msg,
            "prediction": None,
            "confidence": 0.0,
            "all_predictions": []
        }


def preprocess_for_emnist(pil_image, debug=False):

    try:

        if pil_image.mode != 'L':
            pil_image = pil_image.convert('L')


        pil_image = pil_image.resize((28, 28), Image.LANCZOS)

        img_array = np.array(pil_image)

        if debug:
            print(f"Image array shape after resize: {img_array.shape}")
            print(f"Pixel value range: {img_array.min()} - {img_array.max()}")

        img_array = img_array.astype('float32') / 255.0


        if np.mean(img_array) > 0.5:
            img_array = 1.0 - img_array
            if debug:
                print("Inverted image (white bg -> black bg)")


        img_array = img_array.reshape(1, 28, 28, 1)

        if debug:
            print(f"Final preprocessed shape: {img_array.shape}")
            print(f"Final pixel range: {img_array.min():.3f} - {img_array.max():.3f}")

        return img_array

    except Exception as e:
        if debug:
            print(f"Preprocessing error: {str(e)}")
        return None
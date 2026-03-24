import tensorflow as tf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_metadata_to_tflite_model(tflite_model_path, metadata_dict, output_model_path):
    try:
        # Load the TFLite model
        with open(tflite_model_path, 'rb') as f:
            tflite_model = f.read()
        logging.info('Loaded TFLite model: %s', tflite_model_path)

        # Add metadata
        metadata = tf.compat.v1.metadata 
        metadata.add_metadata(tflite_model, metadata_dict)
        logging.info('Metadata added: %s', metadata_dict)

        # Save the updated model
        with open(output_model_path, 'wb') as f:
            f.write(tflite_model)
        logging.info('Saved updated TFLite model: %s', output_model_path)

    except FileNotFoundError as e:
        logging.error('File not found: %s', e)
    except Exception as e:
        logging.error('An error occurred: %s', e)

# Example usage
if __name__ == '__main__':
    # Define the TFLite model path and metadata
    tflite_model_path = 'model.tflite'
    metadata_dict = {'author': 'Shivakrishna6', 'date': '2026-03-24'}
    output_model_path = 'model_with_metadata.tflite'

    # Add metadata to the TFLite model
    add_metadata_to_tflite_model(tflite_model_path, metadata_dict, output_model_path)
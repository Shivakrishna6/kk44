import tensorflow as tf
from tflite_support import metadata
from tflite_support.metadata_writers import python_metadata_writer

# Load the TFLite model
model_file = 'model_fine_tuned.tflite'
model_with_metadata_file = 'model_with_metadata.tflite'

# Load labels
with open('labels_fine_tuned.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize the metadata writer
writer = python_metadata_writer.MetadataWriter.create_for_inference(model_file)

# Create metadata to add
input_metadata = metadata.MetadataTensor(input_tensor_name='input', shape=[1, 224, 224, 3], dtype='float32')
output_metadata = metadata.MetadataTensor(output_tensor_name='output', shape=[1, len(labels)], dtype='float32')

# Set image input normalization and classification output
writer.set_input(input_metadata)
writer.set_output(output_metadata)

# Add label information
writer.set_label_strings(labels)

# Write model with metadata
writer.save(model_with_metadata_file)
print(f'Model with metadata saved as {model_with_metadata_file}')
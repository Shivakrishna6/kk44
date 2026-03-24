import tensorflow as tf  
from tensorflow import lite  

# Load the TFLite model  
model_file = 'model_fine_tuned.tflite'  

# Load the labels  
labels_file = 'labels_fine_tuned.txt'  

# Read labels  
with open(labels_file, 'r') as f:  
    labels = [line.strip() for line in f.readlines()]  

# Load the model  
interpreter = tf.lite.Interpreter(model_path=model_file)  

# Get input and output tensors  
input_details = interpreter.get_input_details()  
output_details = interpreter.get_output_details()  

# Create a metadata writer  
metadata_writer = lite.MetadataWriter.create()  

# Set input tensor metadata  
metadata_writer.set_input_tensor_metadata(  
    input_index=0,  
    shape=input_details[0]['shape'],  
    dtype=input_details[0]['dtype'],  
    name='input_image'  
)  

# Set output tensor metadata  
metadata_writer.set_output_tensor_metadata(  
    output_index=0,  
    shape=output_details[0]['shape'],  
    dtype=output_details[0]['dtype'],  
    labels=labels  
)  

# Save the model with metadata  
metadata_file = 'model_with_metadata.tflite'  
metadata_writer.save_to_file(metadata_file)  

print(f'Model saved to {metadata_file} with metadata included.')  

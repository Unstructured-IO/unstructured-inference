import os
import json
from unstructured_inference.inference.layout import DocumentLayout
import platform  # Import the platform module to get system information
import time

# Directory where the files are located
input_directory = "sample-docs"

# Get system information (e.g., 'x86', 'arm64')
system_info = platform.machine()

# Create the output directory with system information
output_directory = f"layout_results_{system_info}"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# List all files (excluding directories) in the input directory
all_files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]

# Iterate through each file and analyze its layout
for file_name in all_files:
    # Construct the full path to the input file
    file_path = os.path.join(input_directory, file_name)
    
    # Check if the file ends with ".pdf" (only process PDF files that is not too large)
    if file_name.endswith(".pdf") and file_name not in ["pdf2image-memory-error-test-400p.pdf"]:
        print(f"Analyzing layout for {file_name}...")
        start_time = time.time()
        
        # Analyze the layout of the document
        layout = DocumentLayout.from_file(file_path)
        
        # Record the end time
        end_time = time.time()
        
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        
        # Print the time taken to analyze the layout
        print(f"Layout analysis for {file_name} completed in {elapsed_time:.2f} seconds.")
        
        # Construct the output JSON file path for storing the layout results (using the original file name)
        output_json_file = os.path.join(output_directory, f"{os.path.splitext(file_name)[0]}_layout.json")
        
        # Create a list to store the layout elements with only "text" and "type" fields
        elements_dict_list = []
        
        for page in layout.pages:
            for element in page.elements:
                element_dict = {
                    "text": element.text,
                    "type": element.type
                }
                elements_dict_list.append(element_dict)
        
        # Write the layout elements to the output JSON file
        with open(output_json_file, "w", encoding="utf-8") as json_file:
            json.dump(elements_dict_list, json_file, ensure_ascii=False, indent=4)
        
        print(f"Layout analysis for {file_name} completed. Results saved to {output_json_file}")

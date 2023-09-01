import os
from unstructured_inference.inference.layout import DocumentLayout
import platform  # Import the platform module to get system information

# Directory where the files are located
input_directory = "sample-docs"

# Get system information (e.g., 'x86', 'arm64')
system_info = platform.machine()

# Create the output directory with system information
output_directory = f"layout_results_{system_info}"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# List all files in the input directory
all_files = [f for f in os.listdir(input_directory)]

# Iterate through each file and analyze its layout
for file_name in all_files:
    # Construct the full path to the input file
    file_path = os.path.join(input_directory, file_name)
    
    try:
        print(f"Starting Layout analysis for {file_name}...")
        # Analyze the layout of the document
        layout = DocumentLayout.from_file(file_path)
        
        # Construct the output file path for storing the layout results
        output_file = os.path.join(output_directory, f"{file_name}_layout.json")
        
        # Create a list to store the layout elements as dictionaries
        elements_dict_list = []
        
        for page in layout.pages:
            for element in page.elements:
                elements_dict_list.append(element.to_dict())
        
        # Write the layout elements to the output JSON file
        with open(output_json_file, "w", encoding="utf-8") as json_file:
            json.dump(elements_dict_list, json_file)

        print(f"Layout analysis for {file_name} completed. Results saved to {output_file}")
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")

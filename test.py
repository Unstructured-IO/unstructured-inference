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
        output_file = os.path.join(output_directory, f"{os.path.splitext(file_name)[0]}_layout.txt")
        
        # Write the layout output to the output file
        with open(output_file, "w", encoding="utf-8") as output_file:
            for page in layout.pages:
                for element in page.elements:
                    output_file.write(element.text + "\n")

        print(f"Layout analysis for {file_name} completed. Results saved to {output_file}")
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")

import os
from unstructured_inference.inference.layout import DocumentLayout
import platform  # Import the platform module to get system information

# Directory where the PDF files are located
pdf_directory = "sample-docs"

# Get system information (e.g., 'x86', 'arm64')
system_info = platform.machine()

# Create the output directory with system information
output_directory = f"layout_results_{system_info}"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# List all the PDF files in the input directory
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]

# Iterate through each PDF file and analyze its layout
for pdf_file in pdf_files:
    # Construct the full path to the input PDF file
    pdf_path = os.path.join(pdf_directory, pdf_file)
    
    # Analyze the layout of the PDF document
    layout = DocumentLayout.from_file(pdf_path)
    
    # Construct the output file path for storing the layout results
    output_file = os.path.join(output_directory, f"{os.path.splitext(pdf_file)[0]}_layout.txt")
    
    # Write the layout output to the output file
    with open(output_file, "w", encoding="utf-8") as output_file:
        for page in layout.pages:
            for element in page.elements:
                output_file.write(element.text)

    print(f"Layout analysis for {pdf_file} completed. Results saved to {output_file}")

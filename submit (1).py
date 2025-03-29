import os
import zipfile

'''
Run this file to create your zip folder for submission.

IMPORTANT: MAKE SURE THAT YOU RUN THIS FILE FROM THE SAME FOLDER AS YOUR ASSIGNMENT FILES

you will be prompted to enter your team number for naming the zip file... pretty please give the correct team number <3

'''





ASSIGNMENT = 4
REQUIRED_FILES = ["models.py"]




team_num = input("Enter your team number: ")

folder_path = os.getcwd()


contributions = "not-found.txt"
while not os.path.exists(os.path.join(folder_path, contributions)):
    contributions = input("Enter the contributions file (.txt, .pdf, .docx) you would like to include in the zip file: ")
    if not os.path.exists(os.path.join(folder_path, contributions)):
        print("Error: The specified file does not exist. Please try again.")
REQUIRED_FILES.append(contributions)

# Check if all required files exist
missing_files = [file for file in REQUIRED_FILES if not os.path.exists(os.path.join(folder_path, file))]
if missing_files:
    print("Error: The following required files are missing:")
    for file in missing_files:
        print(f"- {file}")
    exit(1)

# Create the zip file
zip_filename = f"A{ASSIGNMENT}_{team_num}.zip"
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for file in REQUIRED_FILES:
        file_path = os.path.join(folder_path, file)
        zipf.write(file_path, arcname=file)

print(f"Zip file '{zip_filename}' created successfully with the required files. Please upload this zip as your submission.")

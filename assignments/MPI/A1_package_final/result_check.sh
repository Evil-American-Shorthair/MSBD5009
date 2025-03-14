#!/bin/bash

# Check if a file name is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <output_file_to_compare>"
    exit 1
fi

# File to compare with output.txt
file_to_compare=$1

# Check if the file exists
if [ ! -f "$file_to_compare" ]; then
    echo "File $file_to_compare does not exist."
    exit 1
fi

# Compare the files
if diff output.txt "$file_to_compare" > /dev/null; then
    echo "Your output is correct."
else
    echo "Your output is incorrect, the files are different."
fi
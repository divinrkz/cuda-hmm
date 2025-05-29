#!/bin/bash

echo "Downloading HMM sequence data files..."

mkdir -p tests/data

BASE_URL="https://caltech-cs155.s3.us-east-2.amazonaws.com/sets/set6/data/"

for i in {0..5}; do
    filename="sequence_data${i}.txt"
    filepath="tests/data/${filename}"
    url="${BASE_URL}${filename}"
    
    echo "Downloading ${filename} to tests/data/..."
    
    if command -v curl > /dev/null; then
        curl -o "${filepath}" "${url}"
    elif command -v wget > /dev/null; then
        wget -O "${filepath}" "${url}"
    else
        echo "Error: Neither curl nor wget is available"
        echo "Please download manually from: ${url}"
        echo "Save to: ${filepath}"
        continue
    fi
    
    if [ -f "${filepath}" ]; then
        echo "✓ ${filename} downloaded successfully to tests/data/"
        # Show file size for verification
        file_size=$(ls -lh "${filepath}" | awk '{print $5}')
        echo "  File size: ${file_size}"
    else
        echo "✗ Failed to download ${filename}"
    fi
done

echo ""
echo "Download complete!"
echo "Files saved in: tests/data/"
echo ""
echo "Directory structure:"
ls -la tests/data/

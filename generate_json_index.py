#!/usr/bin/env python3
"""
Generate a static JSON index of all markdown files in the data directory.
"""

import os
import json
import glob
from datetime import datetime

def generate_json_index():
    """Generate a JSON index of all markdown files in the data directory."""
    
    # Get all markdown files in the data directory
    md_files = glob.glob('data/*.md')
    
    # Create a list of file information
    files_info = []
    for file_path in md_files:
        file_name = os.path.basename(file_path)
        # Extract date from filename (assuming format YYYY-MM-DD.md)
        date_str = file_name.replace('.md', '')
        
        try:
            # Validate date format
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Add file info to the list
            files_info.append({
                'name': file_name,
                'date': date_str,
                'path': file_path
            })
        except ValueError:
            # Skip files that don't match the expected date format
            print(f"Skipping {file_name} - doesn't match expected date format YYYY-MM-DD.md")
            continue
    
    # Sort files by date (newest first)
    files_info.sort(key=lambda x: x['date'], reverse=True)
    
    # Group files by year
    files_by_year = {}
    for file_info in files_info:
        year = file_info['date'][:4]  # Extract year from date string
        if year not in files_by_year:
            files_by_year[year] = []
        files_by_year[year].append(file_info)
    
    # Create the final index structure
    index = {
        'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'files': files_info,
        'filesByYear': files_by_year
    }
    
    # Write the index to a JSON file
    with open('data/index.json', 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    
    print(f"Generated index.json with {len(files_info)} files")

if __name__ == "__main__":
    generate_json_index()

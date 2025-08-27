import os
import json
import glob
from datetime import datetime

def generate_split_json_index():
    """
    Generate a main JSON index with a list of years, and separate JSON files for each year's data.
    """
    # Ensure data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created data directory")
        return
    
    # Get all markdown files
    md_files = glob.glob('data/*.md')
    
    # Process all files into a list
    files_info = []
    for file_path in md_files:
        file_name = os.path.basename(file_path)
        date_str = file_name.replace('.md', '')
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            files_info.append({
                'name': file_name,
                'date': date_str,
                'path': file_path
            })
        except ValueError:
            print(f"Skipping {file_name} - doesn't match format YYYY-MM-DD.md")
            continue
            
    # Group files by year
    files_by_year = {}
    for file_info in files_info:
        year = file_info['date'][:4]
        if year not in files_by_year:
            files_by_year[year] = []
        files_by_year[year].append(file_info)

    # --- New Logic: Generate a separate file for each year ---
    for year, files in files_by_year.items():
        # Sort files within the year (newest first)
        files.sort(key=lambda x: x['date'], reverse=True)
        
        year_index_data = {
            'year': year,
            'files': files
        }
        year_file_path = f'data/index_{year}.json'
        with open(year_file_path, 'w', encoding='utf-8') as f:
            json.dump(year_index_data, f, ensure_ascii=False, indent=2)
        print(f"Generated yearly index: {year_file_path}")

    # --- New Logic: Generate the main, lightweight index.json ---
    # Get a sorted list of available years (newest first)
    available_years = sorted(files_by_year.keys(), reverse=True)
    
    main_index = {
        'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'years': available_years
    }
    
    main_index_path = 'data/index.json'
    with open(main_index_path, 'w', encoding='utf-8') as f:
        json.dump(main_index, f, ensure_ascii=False, indent=2)
    
    print(f"Generated main index.json with {len(available_years)} years.")

if __name__ == "__main__":
    generate_split_json_index()
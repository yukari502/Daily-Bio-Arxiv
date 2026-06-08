import os
import glob
from datetime import datetime

def generate_sitemap():
    """
    Generate sitemap.xml for SEO.
    """
    # Ensure data directory exists
    if not os.path.exists('data'):
        print("Data directory not found. Run generate_json_index.py or daily_arxiv spider first.")
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

    base_url = "https://yukari502.github.io/Daily-Bio-Arxiv"
    sitemap_path = 'sitemap.xml'
    
    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    
    # Create the root element
    urlset = ET.Element('urlset')
    urlset.set('xmlns', 'http://www.sitemaps.org/schemas/sitemap/0.9')
    
    # Add the main index page
    url_element = ET.SubElement(urlset, 'url')
    loc = ET.SubElement(url_element, 'loc')
    loc.text = f"{base_url}/"
    changefreq = ET.SubElement(url_element, 'changefreq')
    changefreq.text = 'daily'
    priority = ET.SubElement(url_element, 'priority')
    priority.text = '1.0'
    
    # Add all individual update pages
    files_info.sort(key=lambda x: x['date'], reverse=True)
    
    for file_info in files_info:
        date_str = file_info['date']
        url_element = ET.SubElement(urlset, 'url')
        
        loc = ET.SubElement(url_element, 'loc')
        loc.text = f"{base_url}/?date={date_str}"
        
        lastmod = ET.SubElement(url_element, 'lastmod')
        lastmod.text = date_str
        
        changefreq = ET.SubElement(url_element, 'changefreq')
        changefreq.text = 'never'
        
        priority = ET.SubElement(url_element, 'priority')
        priority.text = '0.8'
        
    # Write to file with pretty print
    xml_str = minidom.parseString(ET.tostring(urlset, encoding='utf-8')).toprettyxml(indent="  ")
    with open(sitemap_path, 'w', encoding='utf-8') as f:
        # minidom adds its own xml declaration, but we can just write it directly
        f.write(xml_str)
        
    print(f"Generated sitemap.xml with {len(files_info) + 1} URLs.")

if __name__ == "__main__":
    generate_sitemap()

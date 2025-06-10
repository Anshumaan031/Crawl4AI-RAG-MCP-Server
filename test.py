import os
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree
import requests
from typing import List

def parse_sitemap(sitemap_url: str) -> List[str]:
    """Parse a sitemap and extract URLs. Handles both remote URLs and local file paths."""
    urls = []
    try:
        # Check if it's a local file path (multiple formats)
        is_local_file = (
            sitemap_url.startswith("file://") or 
            os.path.isfile(sitemap_url) or
            (len(sitemap_url) > 1 and sitemap_url[1] == ':')  # Windows drive letter
        )
        
        if is_local_file:
            # Handle local file path
            if sitemap_url.startswith("file://"):
                # Convert file:// URL to local path
                sitemap_path = urlparse(sitemap_url).path
                # Fix Windows path (remove leading slash)
                if os.name == 'nt' and sitemap_path.startswith('/') and ':' in sitemap_path:
                    sitemap_path = sitemap_path[1:]
            else:
                # Direct file path
                sitemap_path = sitemap_url
            
            # Convert to Path object for better handling
            sitemap_path = Path(sitemap_path)
            
            print(f"Attempting to read local sitemap file: {sitemap_path}")
            
            if sitemap_path.exists() and sitemap_path.is_file():
                try:
                    with open(sitemap_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        tree = ElementTree.fromstring(content)
                        print(f"Successfully parsed local sitemap file")
                except FileNotFoundError:
                    print(f"Error: Sitemap file not found at {sitemap_path}")
                    return urls
                except ElementTree.ParseError as e:
                    print(f"Error parsing sitemap XML: {e}")
                    return urls
                except Exception as e:
                    print(f"Error reading sitemap file: {e}")
                    return urls
            else:
                print(f"Error: Sitemap file does not exist at {sitemap_path}")
                print(f"Current working directory: {os.getcwd()}")
                print(f"File exists check: {sitemap_path.exists()}")
                print(f"Is file check: {sitemap_path.is_file()}")
                return urls
        else:
            # Handle remote URL
            print(f"Fetching remote sitemap: {sitemap_url}")
            resp = requests.get(sitemap_url, timeout=30)
            resp.raise_for_status()
            
            try:
                tree = ElementTree.fromstring(resp.content)
                print(f"Successfully parsed remote sitemap")
            except ElementTree.ParseError as e:
                print(f"Error parsing sitemap XML: {e}")
                return urls
            finally:
                resp.close()
        
        # Try multiple approaches to extract URLs
        # Method 1: With namespace
        try:
            namespace = {'s': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            url_elements = tree.findall('s:url/s:loc', namespace)
            urls = [loc.text.strip() for loc in url_elements if loc.text]
            print(f"Found {len(urls)} URLs using namespace method")
        except Exception as e:
            print(f"Namespace method failed: {e}")
            urls = []
        
        # Method 2: Without namespace (fallback)
        if not urls:
            try:
                # Remove namespace prefixes and search
                for elem in tree.iter():
                    if elem.tag.endswith('}loc') or elem.tag == 'loc':
                        if elem.text and elem.text.strip():
                            urls.append(elem.text.strip())
                print(f"Found {len(urls)} URLs using fallback method")
            except Exception as e:
                print(f"Fallback method failed: {e}")
        
        # Method 3: XPath with default namespace (last resort)
        if not urls:
            try:
                # Register default namespace
                ElementTree.register_namespace('', 'http://www.sitemaps.org/schemas/sitemap/0.9')
                loc_elements = tree.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                urls = [loc.text.strip() for loc in loc_elements if loc.text]
                print(f"Found {len(urls)} URLs using XPath method")
            except Exception as e:
                print(f"XPath method failed: {e}")
        
        # Debug information
        if not urls:
            print("Debug: Sitemap structure analysis:")
            print(f"Root tag: {tree.tag}")
            print(f"Root attributes: {tree.attrib}")
            print("First few child elements:")
            for i, child in enumerate(tree):
                if i < 3:  # Show first 3 children
                    print(f"  Child {i}: tag={child.tag}, attrib={child.attrib}")
                    for j, grandchild in enumerate(child):
                        if j < 2:  # Show first 2 grandchildren
                            print(f"    Grandchild {j}: tag={grandchild.tag}, text={grandchild.text}")
                            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sitemap {sitemap_url}: {e}")
    except Exception as e:
        print(f"Unexpected error parsing sitemap: {e}")
    
    return urls

# Helper function to test local file access
def test_local_sitemap_access(file_path: str):
    """Test function to debug local file access issues"""
    path = Path(file_path)
    print(f"Testing file access for: {file_path}")
    print(f"Resolved path: {path.resolve()}")
    print(f"Exists: {path.exists()}")
    print(f"Is file: {path.is_file()}")
    print(f"Parent directory exists: {path.parent.exists()}")
    print(f"Current working directory: {os.getcwd()}")
    
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()[:200]  # First 200 chars
                print(f"File content preview: {content}")
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        # List files in the expected directory
        try:
            parent_files = list(path.parent.glob('*.xml'))
            print(f"XML files in parent directory: {parent_files}")
        except Exception as e:
            print(f"Error listing parent directory: {e}")
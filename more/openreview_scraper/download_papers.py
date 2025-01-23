import pandas as pd
import requests
import os
from urllib.parse import urlparse
from pathlib import Path

import argparse
import time
from tqdm import tqdm

from datetime import datetime
import openreview
import json

def download_pdfs(csv_path, url_column, output_dir='downloaded_pdfs', delay=3):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Ensure URL column exists
    if url_column not in df.columns:
        raise ValueError(f"Column '{url_column}' not found in CSV file")
    
    failed_downloads = []
    session = requests.Session()

    for idx, url in enumerate(tqdm(df[url_column])):
        try:
            # Skip empty/invalid URLs
            if pd.isna(url) or not url.strip():
                continue
            

            time.sleep(delay)  # Rate limiting delay
            
            # Get filename from URL or use index
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename.endswith('.pdf'):
                filename = f'paper_{idx}.pdf'
            
            output_path = os.path.join(output_dir, filename)
            
            # Download PDF
            response = session.get(url, stream=True)
            response.raise_for_status()
            
            # Check if it's actually a PDF
            if 'application/pdf' not in response.headers.get('content-type', '').lower():
                raise ValueError('URL does not point to a PDF file')
            
            # Save PDF
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Successfully downloaded: {filename}")
            
        except Exception as e:
            failed_downloads.append((url, str(e)))
            print(f"Failed to download {url}: {str(e)}")
    
    # Print summary
    print(f"\nDownload complete. {len(df) - len(failed_downloads)} successful, {len(failed_downloads)} failed.")
    if failed_downloads:
        print("\nFailed downloads:")
        for url, error in failed_downloads:
            print(f"{url}: {error}")


def download_reviews(csv_path, url_column, output_dir='downloaded_reviews', delay=3):
    """
    Download reviews for papers from OpenReview using paper IDs from a CSV file
    
    Args:
        csv_path: Path to CSV file containing paper IDs
        paper_id_column: Name of column containing OpenReview paper IDs
        output_dir: Directory to save reviews
        delay: Delay between requests in seconds
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Ensure paper ID column exists
    if url_column not in df.columns:
        raise ValueError(f"Column '{url_column}' not found in CSV file")
    
    failed_downloads = []
    client = openreview.Client(baseurl='https://api.openreview.net')
    
    for idx, paper_link in enumerate(tqdm(df[url_column])):
        
        filename = df.iloc[idx]["pdf"]
        
        # print(df.iloc[idx]["pdf"])

        paper_id = paper_link.split("id=")[-1]

        try:
            # Skip empty/invalid paper IDs
            if pd.isna(paper_id) or not str(paper_id).strip():
                continue
                
            time.sleep(delay)  # Rate limiting delay
            
            # Get paper details and reviews
            notes = client.get_notes(forum=paper_id)
            reviews = [note for note in notes if 'Official_Review' in note.invitation]
            paper = client.get_note(paper_id)
            
            paper_data = {
                'title': paper.content['title'],
                'authors': paper.content.get('authors', []),
                'forum_id': paper.forum,
                'reviews': []
            }
            
            for review in reviews:
                review_data = {
                    'id': review.id,
                    'reviewer': review.signatures[0],
                    'rating': review.content['rating'] if 'rating' in review.content else 'N/A',
                    'confidence': review.content['confidence'] if 'confidence' in review.content else 'N/A',
                    'main_review': review.content['main_review'] if 'main_review' in review.content else 'N/A',
                    'date': datetime.fromtimestamp(review.tmdate/1000).strftime('%Y-%m-%d'),
                    'summary': review.content['summary'] if 'summary' in review.content else 'N/A',
                    # 'limitations': review.content['limitations'] if 'limitations' in review.content else 'N/A',
                    # 'strengths': review.content['strengths'] if 'strengths' in review.content else 'N/A',
                    # 'weaknesses': review.content['weaknesses'] if 'weaknesses' in review.content else 'N/A'
                }
                paper_data['reviews'].append(review_data)
            
            # Create safe filename from paper title
            safe_title = "".join([c for c in paper_data['title'] if c.isalnum() or c.isspace()]).rstrip()
            # filename = f"{safe_title}_{paper_id}.json"
            filename = filename.split("/")[-1]
            filename = filename.replace(".pdf", ".json")
            output_path = Path(output_dir) / filename
            
            print(output_path)
            # Save reviews
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(paper_data, f, ensure_ascii=False, indent=2)
                
            print(f"Successfully downloaded reviews for: {filename}")
            
        except Exception as e:
            failed_downloads.append((paper_id, str(e)))
            print(f"Failed to download reviews for paper {paper_id}: {str(e)}")
    
    # Print summary
    print(f"\nDownload complete. {len(df) - len(failed_downloads)} successful, {len(failed_downloads)} failed.")
    if failed_downloads:
        print("\nFailed downloads:")
        for paper_id, error in failed_downloads:
            print(f"{paper_id}: {error}")
            
    # Save failed downloads to file
    if failed_downloads:
        failed_path = Path(output_dir) / 'failed_downloads.json'
        with open(failed_path, 'w', encoding='utf-8') as f:
            json.dump(failed_downloads, f, indent=2)
        print(f"\nFailed downloads saved to: {failed_path}")

def main():
    parser = argparse.ArgumentParser(description='Download PDFs from URLs listed in a CSV file')
    parser.add_argument('csv_path', help='Path to the CSV file containing PDF URLs')
    parser.add_argument('--outputdir', default='downloaded_pdfs', 
                        help='Directory to save downloaded PDFs (default: downloaded_pdfs)')
    
    args = parser.parse_args()
    # download_pdfs(args.csv_path, "pdf", args.outputdir)
    download_reviews(args.csv_path, "forum", args.outputdir)



if __name__ == '__main__':
    main()


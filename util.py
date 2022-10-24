import json
import wget
import logging
import urllib.error
import os.path

def parse_time(x):
    pat = x.split()
    if len(pat) == 1:
        if pat[0].endswith('min'):
            minutes = pat[0]
            return int(minutes[:-3])/60
        elif pat[0].endswith('h'):
            hours = pat[0]
            return int(hours[:-1])
    elif len(pat) == 2:
        hours, minutes = pat
        return int(hours[:-1]) + int(minutes[:-3])/60
    else:
        raise ValueError("Wrong format")
    

def total_runtime():    
    data = json.load(open("data/in/cinedantan_movies.json"))
    total = sum(parse_time(x['runtime']) for x in data if len(x['runtime'])>0)
    print(f"Total runtime: {total:0.2f} hours")

def get_formats(urls):
    D = {}
    for url in urls:
        if url['format'] in D:
            D[url['format']].append(url['url'])
        else:
            D[url['format']]= [url['url']]
    return D
    
def _download():
    data = json.load(open("data/in/cinedantan_movies.json"))
    for film in data:
        logging.info(f"Processing {film['identifier']}")
        formats = get_formats(film['aoFiles'])
        urls = formats.get('h.264', formats.get('512Kb MPEG4', []))
        for url in urls:
            theurl = f"https://archive.org/download/{film['identifier']}/{url}"
            target = f"data/out/cinedantan/{url}"
            if os.path.exists(target):
                logging.info("File already exists")
            else:
                logging.info(f"Downloading {theurl}")
                try:
                    wget.download(theurl, out=target)
                except urllib.error.HTTPError as e:
                    logging.info(f"Download failed: {e}")

formats = ['h.264', '512Kb MPEG4', '256Kb MPEG4']

def download():
    from internetarchive import get_session
    data = json.load(open("data/in/cinedantan_movies.json"))
    s = get_session()
    for film in data:
        logging.info(f"Processing {film['identifier']}")
        item = s.get_item(film['identifier'])
        file_formats = [ list(item.get_files(formats=[form])) for form in formats ]
        file_formats = [ x for x in file_formats if len(x) > 0]
        if len(file_formats) > 0:
            files = file_formats[0]
            logging.info(f"Downloading {len(files)} files") 
            for file in files:
                logging.info(f"Downloading {file.url}")
                file.download(destdir=f"data/out/cinedantan/{film['identifier']}/")
                    
            
                    
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    download()
    

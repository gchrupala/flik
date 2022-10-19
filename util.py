import json
import wget
import logging

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
    
def download():
    data = json.load(open("data/in/cinedantan_movies.json"))
    for film in data:
        logging.info(f"Processing {film['identifier']}")
        formats = get_formats(film['aoFiles'])
        urls = formats.get('h.264', formats.get('512Kb MPEG4', []))
        for url in urls:
            theurl = f"https://archive.org/download/{film['identifier']}/{url}"
            logging.info(f"Downloading {theurl}")
            wget.download(theurl, out=f"data/out/cinedantan/{url}")
            
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    download()
    

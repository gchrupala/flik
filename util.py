import json

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
    
    
data = json.load(open("data/in/cinedantan_movies.json"))
total = sum(parse_time(x['runtime']) for x in data if len(x['runtime'])>0)
print(f"Total runtime: {total:0.2f} hours")

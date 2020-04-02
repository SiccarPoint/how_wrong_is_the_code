# remember to make an HTTPDigestAuth object!

import requests
from requests.auth import HTTPDigestAuth

# auth = HTTPDigestAuth('User', 'key')

r = requests.get('https://api.github.com/repositories', params={'since': 100},
                 auth=auth)
print(r.json()[0]['commits_url'])

rcommits = requests.get(r.json()[0]['commits_url'][:-6])
print(r.json()[0].keys())

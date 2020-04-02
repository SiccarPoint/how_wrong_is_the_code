# remember to make an HTTPDigestAuth object!

import requests, json
from datetime import datetime
from requests.auth import HTTPDigestAuth

# headers = {'Authorization': 'Bearer KEY'}

q = '''query {
  search(first: 3, type: REPOSITORY, query: "doi.org/") {
    edges {
      node {
        ... on Repository {
          nameWithOwner
          ref(qualifiedName: "master") {
            target {
              ... on Commit {
 #               id
                history(first: 100) {
                  pageInfo {
                    hasNextPage
                  }
                  edges {
                    node {
                      author {
                        name
#                        email
#                        date
                      }
                      pushedDate
                      messageHeadline
#                      oid
                      message
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}'''

r = requests.post('https://api.github.com/graphql', json={'query': q}, headers=headers)
aquired_repos = r.json()['data']['search']['edges']

for rep in aquired_repos:
    rep_data = rep['node']
    name = rep_data['nameWithOwner']
    commit_page_data = rep_data['ref']['target']['history']
    has_next_page = commit_page_data['pageInfo']['hasNextPage']
    commits = commit_page_data['edges']  # this is the list of <=100 commits


def convert_datetime(datetime_str):
    yr = int(datetime_str[:4])
    mo = int(datetime_str[5:7])
    da = int(datetime_str[8:10])
    hr = int(datetime_str[11:13])
    mi = int(datetime_str[14:16])
    se = int(datetime_str[17:19])
    dt = datetime(yr, mo, da, hr, mi, se)
    return dt

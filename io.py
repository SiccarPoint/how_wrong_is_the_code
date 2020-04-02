# remember to make an HTTPDigestAuth object!

import requests, json
from datetime import datetime
from requests.auth import HTTPDigestAuth

q = '''query($first: Int!, $query: String!){
  search(first: $first, type: REPOSITORY, query: $query) {
    edges {
      node {
        ... on Repository {
          nameWithOwner
          createdAt
          pushedAt
          ref(qualifiedName: "master") {
            target {
              ... on Commit {
 #               id
                history(first: 100) {
                  totalCount
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

def get_data(first, query, headers):

    r = requests.post('https://api.github.com/graphql', json = {"query": q, "variables": {"first": first, "query": query}}, headers=headers)
    aquired_repos = r.json()['data']['search']['edges']

    return aquired_repos

def process_aquired_data(aquired_repos):

    for rep in aquired_repos:
        rep_data = rep['node']
        name = rep_data['nameWithOwner']
        creation_date = rep_data['createdAt']
        last_push_date = rep_data['pushedAt']
        commit_page_data = rep_data['ref']['target']['history']
        total_commits = rep_data['ref']['target']['history']['totalCount']
        has_next_page = commit_page_data['pageInfo']['hasNextPage']
        commits = commit_page_data['edges']  # this is the list of <=100 commits

        dt_start = convert_datetime(creation_date)
        dt_last_push = convert_datetime(last_push_date)
        dt_delta = dt_last_push-dt_start

        print(name + ":\t" +str(dt_delta) + "\t" +str(total_commits))
        

        yield rep_data, name, creation_date, last_push_date, commit_page_data, has_next_page, commits

def convert_datetime(datetime_str):
    yr = int(datetime_str[:4])
    mo = int(datetime_str[5:7])
    da = int(datetime_str[8:10])
    hr = int(datetime_str[11:13])
    mi = int(datetime_str[14:16])
    se = int(datetime_str[17:19])
    dt = datetime(yr, mo, da, hr, mi, se)
    return dt

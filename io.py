# remember to make an HTTPDigestAuth object!

import requests, json
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

# remember to make an HTTPDigestAuth object!

import requests, json, re
import numpy as np
from matplotlib.pyplot import plot, figure, show, xlabel, ylabel
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

    r = requests.post('https://api.github.com/graphql',
                      json = {"query": q,
                              "variables": {"first": first, "query": query}},
                      headers=headers)
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

        try:
            dt_start = convert_datetime(creation_date)
            dt_last_push = convert_datetime(last_push_date)
        except TypeError:
            continue
        dt_delta = dt_last_push-dt_start

        print(name + ":\t" +str(dt_delta) + "\t" +str(total_commits))


        yield (rep_data, name, creation_date, last_push_date, commit_page_data,
               has_next_page, commits)

def convert_datetime(datetime_str):
    yr = int(datetime_str[:4])
    mo = int(datetime_str[5:7])
    da = int(datetime_str[8:10])
    hr = int(datetime_str[11:13])
    mi = int(datetime_str[14:16])
    se = int(datetime_str[17:19])
    dt = datetime(yr, mo, da, hr, mi, se)
    return dt


def timedelta_to_days(timedelta):
    t = timedelta.days + timedelta.seconds / 86400.
    return t


def first_commit_dtime(commits, is_next_page, override_next_page=False):
    """
    Returns the time of the first commit, or nothing if not the first batch
    """
    if not is_next_page or override_next_page:
        lastpt = -1
        while 1:
            firsttime = convert_datetime(commits[lastpt]['node']['pushedDate'])
            if firsttime is not None:
                return firsttime
            else:
                lastpt -= 1

def yield_commits_data(commits):
    """
    Takes a list of dicts that is "commits", and yields
    (author_name, pushed_datetime, message_headline, message)
    """
    for c in commits:
        data = c['node']
        author_name = data['author']['name']
        try:
            pushed_datetime = convert_datetime(data['pushedDate'])
        except TypeError:
            continue
        message_headline = data['messageHeadline']
        message = data['message']
        yield author_name, pushed_datetime, message_headline, message


def is_commit_bug(message_headline, message):
    """
    Check if the commit appears to be a bug based on the commit text and the
    keywords: bug, mend, broken

    Note that this won't be able to pick bugs folded in with PRs (...yet)

    Examples
    --------
    >>> message = 'nope'
    >>> header = 'noooooooope'
    >>> is_commit_bug(header, message)
    False
    >>> message1 = 'is this a bug?'
    >>> message2 = 'Broken'
    >>> header1 = 'this mends #501'  # note this fails
    >>> header2 = 'Bugs all over the place'
    >>> is_commit_bug(message1, header)
    True
    >>> is_commit_bug(message2, header)
    True
    >>> is_commit_bug(message, header1)
    False
    >>> is_commit_bug(message1, header2)
    True
    """
    bug1 = r'(^|\W)[Bb]ug($|\W)'
    bug2 = r'(^|\W)[Bb]uggy($|\W)'
    bug3 = r'(^|\W)[Bb]ugs($|\W)'
    mend = r'(^|\W)[Mm]end($|\W)'
    broken = r'(^|\W)[Bb]roken($|\W)'
    allposs = bug1 + '|' + bug2 + '|' + bug3 + '|' + mend + '|' + broken
    found = False
    for mess in (message_headline, message):
        found = re.search(allposs, mess) or found
    return found


# headers = {'Authorization': "Bearer TOKEN_HERE"}

data = get_data(10, "physics", headers)

for (rep_data, name, creation_date, last_push_date, commit_page_data,
     has_next_page, commits) in process_aquired_data(data):
    dtimes = []
    times_bugs_fixed = []
    time_to_bug_fix = []
    commit_rate = []
    bug_fix_rate = []
    last_dtime = None
    last_bug_fix = None
    authors = set()
    # firsttime = first_commit_dtime(commits, False, override_next_page=True)
    for auth, dtime, head, mess in yield_commits_data(commits):
        authors.add(auth)
        isbug = is_commit_bug(head, mess)
        # print(isbug)
        if dtime is not None:
            dtimes.append(dtime)
            if last_dtime is None:
                commit_rate.append(None)
            else:
                commit_rate.append(1./timedelta_to_days(last_dtime - dtime))
            # if isbug:
            #     try:
            #         bug_fix_rate.append(1./(dtime - last_bug_fix).seconds)
            #     except TypeError:  # None
            #         bug_fix_rate.append(None)
            #     times_bugs_fixed.append(dtime)
            #     try:
            #         time_to_bug_fix.append((dtime - firsttime).seconds)
            #     except TypeError:
            #         time_to_bug_fix.append(None)
            #     last_bug_fix = dtime
            last_dtime = dtime

    # creation_dtime = convert_datetime(creation_date)
    first_commit_dtime = dtimes[-1]
    from_start_time = [
        timedelta_to_days(time - first_commit_dtime) for time in dtimes[1:]
    ]
    from_start_time_full = [
        timedelta_to_days(time - first_commit_dtime) for time in dtimes
    ]
    figure(1)
    plot(from_start_time, np.log(commit_rate[1:]), 'x')
    xlabel('commit rate (per day)')
    ylabel('log(time since repo creation (days))')
    figure(2)
    plot(np.log(from_start_time_full), list(range(len(dtimes), 0, -1)))

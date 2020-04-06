# remember to make an HTTPDigestAuth object!

import requests, json, re
import numpy as np
from matplotlib.pyplot import plot, figure, show, xlabel, ylabel, xlim, ylim, bar
from datetime import datetime
from header.header import HEADER
from requests.auth import HTTPDigestAuth

q = '''query($first: Int!, $query: String!, $repo_after: String, $commits_after: String){
  search(first: $first, type: REPOSITORY, query: $query, after: $repo_after) {
    edges {
      node {
        ... on Repository {
          nameWithOwner
          name
          owner {
            login
          }
          createdAt
          pushedAt
          ref(qualifiedName: "master") {
            target {
              ... on Commit {
 #               id
                history(first: 100, after: $commits_after) {
                  totalCount
                  pageInfo {
                    hasNextPage
                    commitsEndCursor: endCursor
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
    pageInfo {
      hasNextPage
      reposEndCursor: endCursor
    }
  }
  rateLimit {
    limit
    cost
    remaining
    resetAt
  }
}
'''

q_single_repo = '''
query ($name: String!, $owner: String!, $commits_after: String) {
  repository(name: $name, owner: $owner) {
    object(expression: "master") {
      ...on Commit {
        history(first: 100, after: $commits_after) {
          totalCount
          pageInfo {
            hasNextPage
            commitsEndCursor: endCursor
          }
          edges {
            node {
              author {
                name
                email
                date
              }
              pushedDate
              messageHeadline
              oid
              message
            }
          }
        }
      }
    }
  }
}
'''


def get_data(first, query, cursor, headers):
    r = requests.post('https://api.github.com/graphql',
                      json = {"query": q,
                              "variables": {
                                  "first": first, "query": query,
                                  "repo_after": cursor
                              }},
                      headers=headers)
    try:
        print("Query cost:", r.json()['data']['rateLimit']['cost'])
    except (TypeError, KeyError):
        print(r.json())
        print("You likely requested too much at once!")
    print("Query limit remaining:", r.json()['data']['rateLimit']['remaining'])
    print("Reset at:", r.json()['data']['rateLimit']['resetAt'])
    try:
        aquired_repos = r.json()['data']['search']['edges']
    except TypeError:  # None means issue with form of return
        print(r.json())
    next_page = bool(r.json()['data']['search']['pageInfo']['hasNextPage'])
    cursor = r.json()['data']['search']['pageInfo']['reposEndCursor']
    return aquired_repos, next_page, cursor


def get_commits_single_repo(name, owner, headers, max_iters=10):
    """Return list of all commits for a single identified repo.
    """
    commits_after = None
    next_page = True
    itercount = 0
    all_commits = []
    while next_page and itercount < max_iters:
        r = requests.post('https://api.github.com/graphql',
                          json = {"query": q_single_repo,
                                  "variables": {
                                      "name": name, "owner": owner,
                                      "commits_after": commits_after
                                  }},
                          headers=headers)
        try:
            commit_info = r.json()['data']['repository']['object']['history']
        except TypeError:
            print(r.json())
        next_page = bool(
            commit_info['pageInfo']['hasNextPage']
        )
        commits_after = commit_info['pageInfo']['commitsEndCursor']
        all_commits += commit_info['edges']
        itercount += 1
    return all_commits


def process_aquired_data(aquired_repos):
    for rep in aquired_repos:
        try:  # incomplete returns will fail with Nones in here, hence exception
            rep_data = rep['node']
            nameowner = rep_data['nameWithOwner']
            name = rep_data['name']
            owner = rep_data['owner']['login']
            creation_date = rep_data['createdAt']
            last_push_date = rep_data['pushedAt']
            commit_page_data = rep_data['ref']['target']['history']
            total_commits = rep_data['ref']['target']['history']['totalCount']
            has_next_page = commit_page_data['pageInfo']['hasNextPage']
            commits = commit_page_data['edges']  # this is the list of <=100 commits
            dt_start = convert_datetime(creation_date)
            dt_last_push = convert_datetime(last_push_date)
        except TypeError:
            continue
        dt_delta = dt_last_push-dt_start

        print(name + ":\t" +str(dt_delta) + "\t" +str(total_commits))


        yield (rep_data, nameowner, name, owner, creation_date, last_push_date,
               commit_page_data, has_next_page, commits, total_commits)


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


def build_commit_and_bug_timelines(commits):
    dtimes = []
    times_bugs_fixed = []
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
            if isbug:
                times_bugs_fixed.append(dtime)
            last_dtime = dtime
    return times_bugs_fixed, dtimes, authors


def build_times_from_first_commit(times_bugs_fixed, dtimes):
    try:
        first_commit_dtime = dtimes[-1]
    except IndexError:  # no commits present
        return None # will trigger TypeError during allocation outside fn
    from_start_time = [
        timedelta_to_days(time - first_commit_dtime) for time in dtimes
    ]
    bug_from_start_time = [
        timedelta_to_days(time - first_commit_dtime)
        for time in times_bugs_fixed
    ]
    return bug_from_start_time, from_start_time


def plot_commit_and_bug_rates(from_start_time, bug_from_start_time):
    figure('cumulative commits, time logged')
    plot(np.log(from_start_time), list(range(len(from_start_time), 0, -1)))
    xlabel('Time (logged days)')
    ylabel('Total commits')
    figure('cumulative commits')
    plot(from_start_time, list(range(len(from_start_time), 0, -1)))
    xlabel('Time (days)')
    ylabel('Total commits')
    figure('cumulative bugs')
    plot(bug_from_start_time + [0, ],
         list(range(len(bug_from_start_time), -1, -1)))
    xlabel('Time (days)')
    ylabel('Total bugs')
    # log - 1 fits would work here if needed
    # form of 1 - exp(kx) may be preferred, as a decay process


if __name__ == "__main__":
    pages = 10
    topic = 'physics'
    bug_find_rate = []  # i.e., per bugs per commit
    total_authors = []
    long_repos = []  # will store [num_commits, name, owner]
    cursor = None  # leave this alone
    for i in range(pages):
        data, next_page, new_cursor = get_data(20, topic, cursor, HEADER)

        for (rep_data, nameowner, name, owner, creation_date,
             last_push_date, commit_page_data, has_next_page,
             commits, total_commits) in process_aquired_data(data):
            if total_commits > 100:
                long_repos.append([total_commits, name, owner])
                continue

            times_bugs_fixed, dtimes, authors = build_commit_and_bug_timelines(
                commits)

            total_authors.append(len(authors))
            try:
                bug_find_rate.append(len(times_bugs_fixed) / len(dtimes))
            except ZeroDivisionError:
                bug_find_rate.append(0.)

            try:
                bug_from_start_time, from_start_time = \
                    build_times_from_first_commit(times_bugs_fixed, dtimes)
            except TypeError:  # no commits present
                continue

            plot_commit_and_bug_rates(from_start_time, bug_from_start_time)
        if next_page:
            cursor = new_cursor
        else:
            break

    print('***')
    for repo in sorted(long_repos)[::-1]:
        print(repo)
    print('***')
    input('Found ' + str(len(long_repos)) + ' long repos. Proceed? [Enter]')

    for count, name, owner in sorted(long_repos)[::-1]:
        print('Reading more commits for ' + owner + '/' + name
              + ', total commits: ' + str(count))
        commits = get_commits_single_repo(name, owner, HEADER, max_iters=10)
        print('Successfully loaded ' + str(len(commits)) + ' commits')
        times_bugs_fixed, dtimes, authors = build_commit_and_bug_timelines(
            commits
        )
        total_authors.append(len(authors))
        try:
            bug_find_rate.append(len(times_bugs_fixed) / len(dtimes))
        except ZeroDivisionError:
            bug_find_rate.append(0.)
        try:
            bug_from_start_time, from_start_time = \
                build_times_from_first_commit(times_bugs_fixed, dtimes)
        except TypeError:  # no commits present
            continue
        plot_commit_and_bug_rates(from_start_time, bug_from_start_time)

    figure('Bug find rate, by project, ascending order')
    plot(sorted(bug_find_rate))
    ylabel('Fraction of all commits finding bugs')
    figure('Total committers vs bug find rate')
    plot(total_authors, bug_find_rate, 'x')
    xlabel('Number of authors committing to code')
    ylabel('Fraction of all commits finding bugs')

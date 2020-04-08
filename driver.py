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
          languages (first: 20) {
            nodes {
              name
            }
          }
          createdAt
          pushedAt
		  object(expression: "master:README.md") {
            ... on Blob {
              text
            }
          }
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
            languages_list = rep_data['languages']['nodes']
            creation_date = rep_data['createdAt']
            last_push_date = rep_data['pushedAt']
            commit_page_data = rep_data['ref']['target']['history']
            total_commits = rep_data['ref']['target']['history']['totalCount']
            readme_text = rep_data['object']['text']
            has_next_page = commit_page_data['pageInfo']['hasNextPage']
            commits = commit_page_data['edges']  # this is the list of <=100 commits
            dt_start = convert_datetime(creation_date)
            dt_last_push = convert_datetime(last_push_date)
            languages = set(lang['name'] for lang in languages_list)
        except TypeError:
            continue
        dt_delta = dt_last_push-dt_start

        print(name + ":\t" +str(dt_delta) + "\t" +str(total_commits))


        yield (rep_data, nameowner, name, owner, creation_date, last_push_date,
               commit_page_data, has_next_page, commits, total_commits,
               languages, readme_text)


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


def look_for_badges(readme_text):
    """
    Searches readme_text for badges. Returns list of found badges.
    These at the moment are coveralls and doi.
    """
    coveralls_str = 'https://coveralls.io/'
    doi_str = 'https://doi.org/'
    badges = set()
    if re.search(coveralls_str, readme_text):
        badges.add('coveralls')
    if re.search(doi_str, readme_text):
        badges.add('doi')
    return badges


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


def calc_event_rate(times_of_events):
    try:
        if not np.isclose(times_of_events[-1], 0.):
            events = np.zeros(len(times_of_events) + 1)
            events[1:] = times_of_events[::-1]
        else:
            events = np.zeros(len(times_of_events))
            events[:] = times_of_events[::-1]
    except IndexError:  # times_of_events is empty
        return np.array([]), 0., 0.
    diffs = np.diff(events)
    rates = 1. / diffs
    return rates, np.median(rates), np.mean(rates)


def plot_commit_and_bug_rates(from_start_time, bug_from_start_time,
                              number_of_authors):
    # figure('cumulative commits, time logged')
    # plot(np.log(from_start_time), list(range(len(from_start_time), 0, -1)))
    # xlabel('Time (logged days)')
    # ylabel('Total commits')

    figure('cumulative commits')
    plot(from_start_time, list(range(len(from_start_time), 0, -1)))
    xlabel('Time (days)')
    ylabel('Total commits')

    figure('cumulative bugs')
    plot(bug_from_start_time + [0, ],
         list(range(len(bug_from_start_time), -1, -1)))
    xlabel('Time (days)')
    ylabel('Total bugs')

    # more people means more commits, and broadly linearly, so
    figure('commits per user')
    plot(from_start_time,
         np.arange(len(from_start_time), 0, -1) / number_of_authors)
    xlabel('Time (days)')
    ylabel('Total commits per author')
    # log - 1 fits would work here if needed
    # form of 1 - exp(kx) may be preferred, as a decay process

    figure('commit rate')
    commit_rates, commit_rate_median, commit_rate_mean = calc_event_rate(
        from_start_time
    )
    plot(sorted(commit_rates), '-')

    figure('bug rate')
    bug_rates, bug_rate_median, bug_rate_mean = calc_event_rate(
        bug_from_start_time
    )
    plot(sorted(bug_rates), '-')
    return commit_rate_median, commit_rate_mean, bug_rate_median, bug_rate_mean


if __name__ == "__main__":
    pages = 10
    topic = 'terrainbento'  # 'physics'
    bug_find_rate = []  # i.e., per bugs per commit
    total_authors = []
    total_commits_per_repo = []
    commit_rate_median_per_repo = []
    commit_rate_mean_per_repo = []
    bug_rate_median_per_repo = []
    bug_rate_mean_per_repo = []
    long_repos = []  # will store [num_commits, name, owner]
    coveralls_count = []
    cursor = None  # leave this alone
    for i in range(pages):
        data, next_page, new_cursor = get_data(20, topic, cursor, HEADER)
        for enum, (
                rep_data, nameowner, name, owner, creation_date,
                last_push_date, commit_page_data, has_next_page,
                commits, total_commits, languages, readme_text
                ) in enumerate(process_aquired_data(data)):
            badges = look_for_badges(readme_text)
            if total_commits > 100:
                long_repos.append([total_commits, name, owner,
                                   languages, badges])
                continue

            times_bugs_fixed, dtimes, authors = build_commit_and_bug_timelines(
                commits)

            total_authors.append(len(authors))
            total_commits_per_repo.append(total_commits)
            try:
                bug_find_rate.append(len(times_bugs_fixed) / len(dtimes))
            except ZeroDivisionError:
                bug_find_rate.append(0.)

            try:
                bug_from_start_time, from_start_time = \
                    build_times_from_first_commit(times_bugs_fixed, dtimes)
            except TypeError:  # no commits present
                continue

            (commit_rate_median, commit_rate_mean,
             bug_rate_median, bug_rate_mean) = plot_commit_and_bug_rates(
                from_start_time, bug_from_start_time, len(authors)
            )
            commit_rate_median_per_repo.append(commit_rate_median)
            commit_rate_mean_per_repo.append(commit_rate_mean)
            bug_rate_median_per_repo.append(bug_rate_median)
            bug_rate_mean_per_repo.append(bug_rate_mean)
            if 'coveralls' in badges:
                coveralls_count.append([enum, owner, name])
        if next_page:
            cursor = new_cursor
        else:
            break

    print('*****')
    for repo in sorted(long_repos)[::-1]:
        print(repo)

    print('*****')
    short_repos = len(commit_rate_mean_per_repo)
    short_count = len(coveralls_count)
    print('Of ' + str(short_repos) + ' short repositories, '
          + str(short_count) + ' use coveralls')
    if short_count > 0:
        print("They are:")
        for ln in coveralls_count:
            print(ln)

    print('***')
    input('Found ' + str(len(long_repos)) + ' long repos. Proceed? [Enter]')

    for enum_long, (
                count, name, owner, languages, badges
            ) in enumerate(sorted(long_repos)[::-1]):
        print('Reading more commits for ' + owner + '/' + name
              + ', total commits: ' + str(count))
        commits = get_commits_single_repo(name, owner, HEADER, max_iters=10)
        print('Successfully loaded ' + str(len(commits)) + ' commits')
        times_bugs_fixed, dtimes, authors = build_commit_and_bug_timelines(
            commits
        )
        total_authors.append(len(authors))
        total_commits_per_repo.append(len(commits))
        try:
            bug_find_rate.append(len(times_bugs_fixed) / len(dtimes))
        except ZeroDivisionError:
            bug_find_rate.append(0.)
        try:
            bug_from_start_time, from_start_time = \
                build_times_from_first_commit(times_bugs_fixed, dtimes)
        except TypeError:  # no commits present
            continue
        commit_rate_median, commit_rate_mean, bug_rate_median, bug_rate_mean = \
            plot_commit_and_bug_rates(from_start_time, bug_from_start_time,
                                      len(authors))
        commit_rate_median_per_repo.append(commit_rate_median)
        commit_rate_mean_per_repo.append(commit_rate_mean)
        bug_rate_median_per_repo.append(bug_rate_median)
        bug_rate_mean_per_repo.append(bug_rate_mean)
        if 'coveralls' in badges:
            coveralls_count.append([enum_long + enum, owner, name])

    print('*****')
    total_repos = len(commit_rate_mean_per_repo)
    long_repos = total_repos - short_repos
    long_count = len(coveralls_count) - short_count
    print('Of ' + str(long_repos) + ' long repositories, '
          + str(long_count) + ' use coveralls')
    print('Of ' + str(total_repos) + ' total repositories, '
          + str(len(coveralls_count)) + ' use coveralls')
    if len(coveralls_count) > 0:
        print(
        "This is a list of all the coveralls repositories, ending with "
        + "the long repositories > 100 commits, listed as "
        + "[ID_in_repo_lists, owner, name]:"
        )
        for ln in coveralls_count:
            print(ln)

    figure('Bug find rate, by project, ascending order')
    plot(sorted(bug_find_rate))
    ylabel('Fraction of all commits finding bugs')
    figure('Total committers vs bug find rate')
    plot(total_authors, bug_find_rate, 'x')
    xlabel('Number of authors committing to code')
    ylabel('Fraction of all commits finding bugs')

    author_numbers = list(set(total_authors))
    median_author_num_commits = []
    mean_author_num_commits = []
    for author_num in author_numbers:
        commits_for_author_num = np.equal(total_authors, author_num)
        median_author_num_commits.append(np.median(
            np.array(total_commits_per_repo)[commits_for_author_num]
        ))
        mean_author_num_commits.append(np.mean(
            np.array(total_commits_per_repo)[commits_for_author_num]
        ))
    figure('commits vs authors')
    plot(total_authors, total_commits_per_repo, 'x')
    plot(author_numbers, median_author_num_commits, 'o')
    xlabel('Number of authors committing to code')
    ylabel('Total number of commits')

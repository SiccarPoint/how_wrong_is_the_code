# remember to make an HTTPDigestAuth object!

import requests, json, re
import numpy as np
from matplotlib.pyplot import plot, figure, show, xlabel, ylabel, xlim, ylim, bar, hist
from datetime import datetime
from header.header import HEADER
from requests.auth import HTTPDigestAuth

COUNT_ADDITIONS = False
# This is a hardwired trigger as doing this makes it very likely we hit the
# API query limiters, requiring a painful decrease in performance &
# unpredictable crashes can occur
if COUNT_ADDITIONS:
    history_page = 50
else:
    history_page = 100


q = '''query($first: Int!, $query: String!, $repo_after: String, $commits_after: String){
  search(first: $first, type: REPOSITORY, query: $query, after: $repo_after) {
    edges {
      node {
        ... on Repository {
          nameWithOwner
          diskUsage
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
                history(first: '''
q += str(history_page)
q += ''', after: $commits_after) {
                  totalCount
                  pageInfo {
                    hasNextPage
                    commitsEndCursor: endCursor
                  }
                  edges {
                    node {\n'''
if COUNT_ADDITIONS:
    q += '                      additions\n'
q += '''                      author {
                        name
#                        email
#                        date
                      }
                      pushedDate
                      messageHeadline
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
'''
if COUNT_ADDITIONS:
    q_single_repo += '''
              additions
'''
q_single_repo += '''
              author {
                name
                email
                date
              }
              pushedDate
              messageHeadline
              message
            }
          }
        }
      }
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
    # once this is done, print where we are in the API costings:
    print("Query cost:", r.json()['data']['rateLimit']['cost'])
    print("Query limit remaining:",
          r.json()['data']['rateLimit']['remaining'])
    print("Reset at:", r.json()['data']['rateLimit']['resetAt'])
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
            commits = commit_page_data['edges']
            # ^^this is the list of <=long_repo commits
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
    (author_name, pushed_datetime, message_headline, message, additions)
    """
    for c in commits:
        data = c['node']
        author_name = data['author']['name']
        if COUNT_ADDITIONS:
            additions = data['additions']
        else:
            additions = None
        try:
            pushed_datetime = convert_datetime(data['pushedDate'])
        except TypeError:
            continue
        message_headline = data['messageHeadline']
        message = data['message']
        yield (author_name, pushed_datetime, message_headline, message,
               additions)


def is_commit_bug(message_headline, message):
    """
    Check if the commit appears to be a bug based on the commit text and the
    keywords: bug, mend, broken, forgot, work(s) right/correctly, deal(s) with,
    typo, wrong, fix(es)
    (established from reviewing project commit logs)

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
    # regex syntax reminder: ^ start of line, $ end of line, \W not word char
    # s? s zero or one time
    bug1 = r'(^|\W)[Bb]ug($|\W)'
    bug2 = r'(^|\W)[Bb]uggy($|\W)'
    bug3 = r'(^|\W)[Bb]ugs($|\W)'
    mend = r'(^|\W)[Mm]end($|\W)'
    broken = r'(^|\W)[Bb]roken($|\W)'
    forgot = r'(^|\W)[Ff]orgot($|\W)'
    worksright = r'(^|\W)works? right($|\W)'
    workscorrectly = r'(^|\W)works? correctly($|\W)'
    dealwith = r'(^|\W)[Dd]eals? with($|\W)'
    typo = r'(^|\W)[Tt]ypo($|\W)'
    wrong = r'(^|\W)[Ww]rong($|\W)'
    fix = r'(^|\W)[Ff]ixe?s?($|\W)'
    allposs = (
        bug1 + '|' + bug2 + '|' + bug3 + '|' + mend + '|' + broken + '|'
        + forgot + '|' + worksright + '|' + workscorrectly + '|'
        + dealwith + '|' + typo + '|' + wrong + '|' + fix
    )
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
    additions = []
    # firsttime = first_commit_dtime(commits, False, override_next_page=True)
    for auth, dtime, head, mess, adds in yield_commits_data(commits):
        authors.add(auth)
        isbug = is_commit_bug(head, mess)
        # print(isbug)
        if dtime is not None:
            dtimes.append(dtime)
            if isbug:
                times_bugs_fixed.append(dtime)
            last_dtime = dtime
            if COUNT_ADDITIONS:
                additions.append(adds)
    return times_bugs_fixed, dtimes, authors, additions


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
                              number_of_authors, additions,
                              highlight=False):

    if highlight:
        fmt = 'k'
    else:
        fmt = ''
    # figure('cumulative commits, time logged')
    # plot(np.log(from_start_time), list(range(len(from_start_time), 0, -1)))
    # xlabel('Time (logged days)')
    # ylabel('Total commits')

    figure('cumulative commits')
    plot(from_start_time, list(range(len(from_start_time), 0, -1)), fmt)
    xlabel('Time (days)')
    ylabel('Total commits')

    figure('cumulative bugs')
    plot(bug_from_start_time + [0, ],
         list(range(len(bug_from_start_time), -1, -1)), fmt + 'x-')
    xlabel('Time (days)')
    ylabel('Total bugs')

    figure('cumulative bugs vs cumulative commits')
    num_commits_at_each_bug = np.searchsorted(from_start_time[::-1],
                                              [0, ] + bug_from_start_time[::-1])
    plot(num_commits_at_each_bug, list(range(len(num_commits_at_each_bug))))
    xlabel('Cumulative number of commits')
    ylabel('Cumulative number of bugs')

    if COUNT_ADDITIONS:
        figure('cumulative bugs vs cumulative code line additions')
        # note this ONLY assesses additions, as fresh code is more likely to
        # introduce new bugs than modified code
        cumulative_lines = np.cumsum(additions[::-1])
        plot(cumulative_lines[num_commits_at_each_bug],
             list(range(len(num_commits_at_each_bug))))
        xlabel('Cumulative number of added lines')
        ylabel('Cumulative number of bugs')
        # note this creates POORLY linear, very steppy plots, notably much
        # worse than the bugs vs commits plot

    # more people means more commits, and broadly linearly, so
    figure('commits per user')
    plot(from_start_time,
         np.arange(len(from_start_time), 0, -1) / number_of_authors, fmt)
    xlabel('Time (days)')
    ylabel('Total commits per author')
    # log - 1 fits would work here if needed
    # form of 1 - exp(kx) may be preferred, as a decay process

    #figure('commit rate')
    commit_rates, commit_rate_median, commit_rate_mean = calc_event_rate(
        from_start_time
    )
    #plot(sorted(commit_rates), fmt + '-')

    #figure('bug rate')
    bug_rates, bug_rate_median, bug_rate_mean = calc_event_rate(
        bug_from_start_time
    )
    # plot(sorted(bug_rates), fmt + 'x-')
    return commit_rate_median, commit_rate_mean, bug_rate_median, bug_rate_mean


def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


if __name__ == "__main__":
    pages = 20  # 20
    max_iters_for_commits = 50
    topic = 'physics'  # 'landlab', 'terrainbento', 'physics', 'chemistry', 'doi.org'
    # the search for Landlab isn't pulling landlab/landlab as a long repo!? Check
    if COUNT_ADDITIONS:
        # not yet quite stable
        get_data_limit = 10
        long_repo = 50
    else:
        get_data_limit = 20
        long_repo = 100

    print('Searching on ' + topic)
    bug_find_rate = []  # i.e., per bugs per commit
    total_authors = []
    total_bugs_per_repo = []
    commit_rate_median_per_repo = []
    commit_rate_mean_per_repo = []
    bug_rate_median_per_repo = []
    bug_rate_mean_per_repo = []
    long_repos = []  # will store [num_commits, name, owner]
    coveralls_count = []
    total_commits_from_API = []
    cursor = None  # leave this alone
    for i in range(pages):
        data, next_page, new_cursor = get_data(get_data_limit, topic, cursor,
                                               HEADER)
        for enum, (
                rep_data, nameowner, name, owner, creation_date,
                last_push_date, commit_page_data, has_next_page,
                commits, total_commits, languages, readme_text
                ) in enumerate(process_aquired_data(data)):
            badges = look_for_badges(readme_text)
            if total_commits > long_repo:
                long_repos.append([total_commits, name, owner,
                                   languages, badges, total_commits])
                continue

            times_bugs_fixed, dtimes, authors, additions = \
                build_commit_and_bug_timelines(commits)

            total_authors.append(len(authors))
            # note this may separate out same author with different IDs, e.g,
            # Katherine Barnhart vs Katy Barnhart vs kbarnhart
            # Not much we can do about this; hope it comes out in the wash
            total_bugs_per_repo.append(len(times_bugs_fixed))
            total_commits_from_API.append(total_commits)
            try:
                bug_find_rate.append(len(times_bugs_fixed) / len(dtimes))
            except ZeroDivisionError:
                bug_find_rate.append(0.)

            try:
                bug_from_start_time, from_start_time = \
                    build_times_from_first_commit(times_bugs_fixed, dtimes)
            except TypeError:  # no commits present
                continue

            if 'coveralls' in badges:
                coveralls_count.append([enum, owner, name])
                emph = True
            else:
                emph = False
            (commit_rate_median, commit_rate_mean,
             bug_rate_median, bug_rate_mean) = plot_commit_and_bug_rates(
                from_start_time, bug_from_start_time, len(authors),
                additions, emph
            )
            commit_rate_median_per_repo.append(commit_rate_median)
            commit_rate_mean_per_repo.append(commit_rate_mean)
            bug_rate_median_per_repo.append(bug_rate_median)
            bug_rate_mean_per_repo.append(bug_rate_mean)
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
    print('Found ' + str(len(long_repos)) + ' long repos.')
    # input('Proceed? [Enter]')

    for enum_long, (
                count, name, owner, languages, badges, total_commits
            ) in enumerate(sorted(long_repos)[::-1]):
        print('Reading more commits for ' + owner + '/' + name
              + ', total commits: ' + str(count))
        commits = get_commits_single_repo(name, owner, HEADER,
                                          max_iters=max_iters_for_commits)
        print('Successfully loaded ' + str(len(commits)) + ' commits')
        times_bugs_fixed, dtimes, authors, additions = \
            build_commit_and_bug_timelines(commits)
        total_authors.append(len(authors))
        total_bugs_per_repo.append(len(times_bugs_fixed))
        total_commits_from_API.append(total_commits)
        try:
            bug_find_rate.append(len(times_bugs_fixed) / len(dtimes))
        except ZeroDivisionError:
            bug_find_rate.append(0.)
        try:
            bug_from_start_time, from_start_time = \
                build_times_from_first_commit(times_bugs_fixed, dtimes)
        except TypeError:  # no commits present
            continue
        if 'coveralls' in badges:
            coveralls_count.append([enum_long + short_repos, owner, name])
            emph = True
        else:
            emph = False
        commit_rate_median, commit_rate_mean, bug_rate_median, bug_rate_mean = \
            plot_commit_and_bug_rates(from_start_time, bug_from_start_time,
                                      len(authors), additions, emph)
        commit_rate_median_per_repo.append(commit_rate_median)
        commit_rate_mean_per_repo.append(commit_rate_mean)
        bug_rate_median_per_repo.append(bug_rate_median)
        bug_rate_mean_per_repo.append(bug_rate_mean)

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
        + "the long repositories > long_repo commits, listed as "
        + "[ID_in_repo_lists, owner, name]:"
        )
        for ln in coveralls_count:
            print(ln)

    cov_indices = [cov[0] for cov in coveralls_count]

    author_numbers = list(set(total_authors))
    median_author_num_commits = []
    mean_author_num_commits = []
    median_author_bug_fraction = []
    mean_author_bug_fraction = []
    for author_num in author_numbers:
        commits_for_author_num = np.equal(total_authors, author_num)
        median_author_num_commits.append(np.median(
            np.array(total_commits_from_API)[commits_for_author_num]
        ))
        median_author_bug_fraction.append(np.median(
            np.array(bug_find_rate)[commits_for_author_num]
        ))
        mean_author_num_commits.append(np.mean(
            np.array(total_commits_from_API)[commits_for_author_num]
        ))
        mean_author_bug_fraction.append(np.mean(
            np.array(bug_find_rate)[commits_for_author_num]
        ))

    figure('commits vs authors')
    plot(total_authors, total_commits_from_API, 'x')
    plot(author_numbers, median_author_num_commits, 'o')
    plot(np.array(total_authors)[cov_indices],
         np.array(total_commits_from_API)[cov_indices], 'kx')
    xlabel('Number of authors committing to code')
    ylabel('Total number of commits')

    figure('bugs vs authors')
    plot(total_authors, total_bugs_per_repo, 'x')
    plot(np.array(total_authors)[cov_indices],
         np.array(total_bugs_per_repo)[cov_indices], 'kx')
    xlabel('Number of authors committing to code')
    ylabel('Total number of bugs')

    # figure('Bug find fraction, by project, ascending order')
    # plot(sorted(bug_find_rate))
    # ylabel('Fraction of all commits finding bugs')

    # now as more like an actual CDF
    figure('bug find fraction as CDF')
    bug_commit_fraction_in_order = np.array(sorted(bug_find_rate))
    CDF_count = np.arange(bug_commit_fraction_in_order.size, dtype=float)
    CDF_count /= CDF_count.max()
    plot(bug_commit_fraction_in_order, CDF_count)
    ylabel('Cumulative probability density')
    xlabel('Fraction of all commits finding bugs')

    figure('Bug find fraction, histogram')
    hist(bug_find_rate, bins='auto')
    hist(np.array(bug_find_rate)[np.array(bug_find_rate) > 0], bins='auto',
         alpha=0.5)
    xlabel('Fraction of all commits finding bugs')

    figure('Total committers vs bug fraction rate')
    bug_find_rate_array = np.array(bug_find_rate)
    total_authors_array = np.array(total_authors)
    plot(total_authors, bug_find_rate, 'x')
    plot(author_numbers, median_author_bug_fraction, 'o')
    plot(total_authors_array[cov_indices],
         bug_find_rate_array[cov_indices], 'kx')
    xlabel('Number of authors committing to code')
    ylabel('Fraction of all commits finding bugs')

    figure('Total commits vs bug fraction rate')
    total_commits_from_API_array = np.array(total_commits_from_API)
    total_commits_order = np.argsort(total_commits_from_API_array)
    total_commits_IN_order = total_commits_from_API_array[total_commits_order]
    bug_find_rate_ordered = bug_find_rate_array[total_commits_order]
    bug_find_rate_moving_avg = moving_average(bug_find_rate_ordered, n=20)

    plot(total_commits_from_API, bug_find_rate, 'x')
    plot(total_commits_IN_order[10:-9], bug_find_rate_moving_avg, '-')
    plot(total_commits_from_API_array[cov_indices],
         bug_find_rate_array[cov_indices], 'kx')
    xlabel('Total number of commits')
    ylabel('Fraction of all commits finding bugs')

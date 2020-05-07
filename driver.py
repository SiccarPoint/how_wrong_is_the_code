# remember to make an HTTPDigestAuth object!

import requests, json, re, os, pandas, sqlite3, time, sqlalchemy
import numpy as np
from matplotlib.pyplot import plot, figure, show, xlabel, ylabel, xlim, ylim, bar, hist
from datetime import datetime
from header.header import HEADER
from requests.auth import HTTPDigestAuth
from copy import copy
from sqlalchemy.exc import OperationalError, DatabaseError
from utils import moving_average

COUNT_ADDITIONS = True
# This is a hardwired trigger as doing this makes it very likely we hit the
# API query limiters, requiring a painful decrease in performance &
# unpredictable crashes can occur
if COUNT_ADDITIONS:
    HISTORY_PAGE = 25
else:
    HISTORY_PAGE = 100

LANGUAGES_TO_TEST_FOR = (
    'Java', 'C', 'C++', 'Python', 'Cython', 'C#', 'Ruby', 'MATLAB',
    'Objective-C', 'R', 'Fortran 77', 'Fortran 90', 'Fortran 95', 'Rust',
    'Haskell'
)

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
q += str(HISTORY_PAGE)
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
        print("Query limit remaining:",
              r.json()['data']['rateLimit']['remaining'])
        print("Reset at:", r.json()['data']['rateLimit']['resetAt'])
        aquired_repos = r.json()['data']['search']['edges']
    except TypeError:  # None means issue with form of return
        print(r.json())
        return cursor  # so we can repeat this call in a different query
    next_page = bool(r.json()['data']['search']['pageInfo']['hasNextPage'])
    cursor = r.json()['data']['search']['pageInfo']['reposEndCursor']
    return aquired_repos, next_page, cursor


def get_commits_single_repo(name, owner, headers, max_iters=10,
                            query_fail_repeats=5):
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
        except TypeError:  # query failed
            repeat_count = 0
            while repeat_count < query_fail_repeats:
                time.sleep(30. + 10. * np.random.rand())  # cooldown period
                r = requests.post('https://api.github.com/graphql',
                                  json = {"query": q_single_repo,
                                          "variables": {
                                              "name": name, "owner": owner,
                                              "commits_after": commits_after
                                          }},
                                  headers=headers)
                try:
                    commit_info = \
                        r.json()['data']['repository']['object']['history']
                except TypeError:
                    repeat_count += 1
                    continue
                else:
                    break
            else:
                raise TypeError("Query failed repeatedly, aborting")
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
        rep_data = rep['node']
        nameowner = rep_data['nameWithOwner']
        name = rep_data['name']
        owner = rep_data['owner']['login']
        languages_list = rep_data['languages']['nodes']
        creation_date = rep_data['createdAt']
        last_push_date = rep_data['pushedAt']
        try:
            commit_page_data = rep_data['ref']['target']['history']
        except TypeError:
            # repo returns broken commit data, so
            print('***Broken commit interface for',  nameowner)
            continue
        total_commits = rep_data['ref']['target']['history']['totalCount']
        has_next_page = commit_page_data['pageInfo']['hasNextPage']
        commits = commit_page_data['edges']
        try:
            readme_text = rep_data['object']['text']
        except TypeError:
            # no readme file, so
            readme_text = ''
        # ^^this is the list of <=long_repo commits
        dt_start = convert_datetime(creation_date)
        dt_last_push = convert_datetime(last_push_date)
        languages = set(lang['name'] for lang in languages_list)
        # except TypeError:
        #     continue
        dt_delta = dt_last_push-dt_start

        print(name + ":\t" +str(dt_delta) + "\t" +str(total_commits))


        yield (rep_data, nameowner, name, owner, creation_date, last_push_date,
               commit_page_data, has_next_page, commits, total_commits,
               languages, readme_text)


def get_process_save_data_all_repos(calls, first, query, long_repo_length,
                                    cursor, headers, continue_run=True,
                                    query_fail_repeats=5):
    """
    Operates get_data and process_aquired_data to produce the data
    ingested by the other functions, but then saves it rather than outputting
    it. At the end of the run, we save a cursor as to permit continuing
    (save_cursor.json).

    Parameters
    ----------
    calls : int
        Number of times to call the API in total
    first : int
        Entries per query to the API (needs tuning to data requested). Total
        repos queried is then technically calls * first, but the API often
        seems to throttle the call, such that less than first entries are
        returned.
    query : str
        The search term to use
    cursor : str or None
        Cursor for the starting point of the search (None if the start)
    headers : str
        Your Security Key for the Github API (DO NOT SAVE IN THIS SCRIPT)
    continue_run : bool
        If True, looks for a cursor savefile (savecursor.json), loads
        it, and uses it as the key from which to continue.
    query_fail_repeats : int
        Number of repeat attempts permitted after a failed API call.
    """
    if not continue_run:
        try:
            os.mkdir(query_to_queryfname(query))
        except FileExistsError:
            raise FileExistsError('A save already exists for this query! '
                                  + 'Delete its directory to create a new one.')
        data_for_repo_short = {}
        data_for_repo_long = {}
    else:
        print('Continuing existing search...')
        with open(os.path.join(query_to_queryfname(query), 'savecursor.json'), 'r') as infile:
            cursor = json.load(infile)
        with open(os.path.join(query_to_queryfname(query), 'savedata_short.json'), 'r') as infile:
            data_for_repo_short = json.load(infile)
        with open(os.path.join(query_to_queryfname(query), 'savedata_long.json'), 'r') as infile:
            data_for_repo_long = json.load(infile)
    for i in range(calls):
        get_data_out = get_data(first, query, cursor, headers)
        if len(get_data_out) != 3:
            repeat_count = 0
            while repeat_count < query_fail_repeats:
                time.sleep(30. + 10. * np.random.rand())  # cooldown period
                get_data_out = get_data(first, query, cursor, headers)
                if len(get_data_out) == 3:
                    break
                else:
                    assert get_data_out == cursor  # check no advance
                    repeat_count += 1
                    # ...and loop continues
            else:
                raise TypeError("Query failed repeatedly, aborting")
        aquired_repos, next_page, cursor = get_data_out
        for (
            rep_data, nameowner, name, owner, creation_date,
            last_push_date, commit_page_data, has_next_page, commits,
            total_commits, languages, readme_text
        ) in process_aquired_data(aquired_repos):
            if total_commits < long_repo_length:
                data_for_repo_short[nameowner] = {
                    'rep_data': rep_data,
                    'name':name,
                    'owner':owner,
                    'creation_date':creation_date,
                    'last_push_date':last_push_date,
                    'commit_page_data':commit_page_data,
                    'has_next_page': has_next_page,
                    'commits': commits,
                    'total_commits': total_commits,
                    'languages': list(languages),
                    'readme_text': readme_text
                }
            else:
                data_for_repo_long[nameowner] = {
                    'rep_data': rep_data,
                    'name':name,
                    'owner':owner,
                    'creation_date':creation_date,
                    'last_push_date':last_push_date,
                    'commit_page_data':commit_page_data,
                    'has_next_page': has_next_page,
                    'commits': commits,
                    'total_commits': total_commits,
                    'languages': list(languages),
                    'readme_text': readme_text
                }
            # these only contains basic Python types, so saving should be OK
    with open(os.path.join(query_to_queryfname(query), 'savedata_short.json'), 'w') as outfile:
        json.dump(data_for_repo_short, outfile)
    with open(os.path.join(query_to_queryfname(query), 'savedata_long.json'), 'w') as outfile:
        json.dump(data_for_repo_long, outfile)
    with open(os.path.join(query_to_queryfname(query), 'savecursor.json'), 'w') as outfile:
        json.dump(cursor, outfile)
    # these all overwrite any existing content


def get_process_save_commit_data_long_repos(query, headers, max_iters):
    """
    This func uses get_commits_single_repo to interrogate the GitHub API for
    max_iters pages of commits for the long repositories specified in an
    existing savefile, created the the search term query. Commit savefile is
    "query/savedata_long_commits.json".
    If the API call fails, function will print a warning message but still
    save an output file to allow continuing of the search in a subsequent call.

    Parameters
    ----------
    query : str
        The search term(s) used to create the savefile of long repos
    headers : str
        Your GitHub authentification token (DO NOT SAVE IN THIS FILE)
    max_iters : int
        The number of pages of commits to read.
    """
    # See if there is an existing dict to load:
    try:
        with open(os.path.join(query_to_queryfname(query), 'savedata_long_commits.json'),
                  'r') as infile:
            long_repo_commit_dict = json.load(infile)
        print('Long commit loader is adding to an existing savefile...')
    except FileNotFoundError:
        long_repo_commit_dict = {}
    repo_count = 0
    for (
        rep_data, nameowner, name, owner, creation_date,
        last_push_date, commit_page_data, has_next_page,
        commits, total_commits, languages, readme_text
    ) in load_processed_data_all_repos(query, 'long'):
        if nameowner not in long_repo_commit_dict.keys():
            print('Calling API for', nameowner)
            print('This is long repo', repo_count)
            try:
                commits = get_commits_single_repo(name, owner, headers,
                                                  max_iters=max_iters)
            except TypeError:  # API refuses connection repeatedly
                print('WARNING: get_process_save_commit_data_long_repos did '
                      + 'not complete. Run it again to finish off remaining '
                      + 'repos.')
                continue
            long_repo_commit_dict[nameowner] = commits
        else:
            print('Skipping', nameowner)
        repo_count += 1
    with open(os.path.join(query_to_queryfname(query), 'savedata_long_commits.json'),
              'w') as outfile:
        json.dump(long_repo_commit_dict, outfile)
    # this overwrites, if there was existing content


def load_processed_data_all_repos(query, short_or_long_repos):
    """
    Loads the output from get_process_save_data_all_repos.

    Parameters
    ----------
    query : str
        The search term(s) used to create the save.
    short_or_long_repos : str in ('short', 'long')
        Whether to load the dict for the short repos, or the long ones
        (total_commits >= long_repo).
    """
    if short_or_long_repos == 'short':
        savefile = 'savedata_short.json'
    elif short_or_long_repos == 'long':
        savefile = 'savedata_long.json'
    else:
        raise ValueError("short_or_long_repos must be 'short' or 'long'")
    with open(os.path.join(query_to_queryfname(query), savefile)) as json_file:
        data_from_repo = json.load(json_file)
    print('Loaded', len(data_from_repo), short_or_long_repos, 'repositories.')
    for nameowner, repo_dict in data_from_repo.items():
        rep_data = repo_dict['rep_data']
        name = repo_dict['name']
        owner = repo_dict['owner']
        creation_date = repo_dict['creation_date']
        last_push_date = repo_dict['last_push_date']
        commit_page_data = repo_dict['commit_page_data']
        has_next_page = repo_dict['has_next_page']
        commits = repo_dict['commits']
        total_commits = repo_dict['total_commits']
        languages = set(repo_dict['languages'])
        readme_text = repo_dict['readme_text']
        yield (rep_data, nameowner, name, owner, creation_date,
               last_push_date, commit_page_data, has_next_page, commits,
               total_commits, languages, readme_text)


def load_processed_commit_data_long_repos(query):
    with open(os.path.join(query_to_queryfname(query), 'savedata_long_commits.json'),
              'r') as infile:
        long_repo_commit_dict = json.load(infile)
    return long_repo_commit_dict


def query_to_queryfname(query):
    """Strip bad chars (/ :) from a query to use it as a filename."""
    fname = query
    for ch in ('/', ':'):
        fname = fname.replace(ch, '')
    return fname


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
            firsttime = convert_datetime(commits[lastpt]['node']['author']['date'])
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
        try:
            author_name = data['author']['name']
        except TypeError:
            # This commit is recorded entirely wrongly, so
            continue
        if COUNT_ADDITIONS:
            additions = data['additions']
        else:
            additions = None
        pushed_datetime = convert_datetime(data['author']['date'])
        message_headline = data['messageHeadline']
        message = data['message']
        yield (author_name, pushed_datetime, message_headline, message,
               additions)


def is_commit_bug(search_type, message_headline, message):
    """
    Check if the commit appears to be a bug based on the commit text and the
    keywords: bug, mend, & variants; broken; forgot; work(s) right/correctly;
    deal(s) with; typo; wrong; fix(es).
    (established from reviewing project commit logs)

    Note that this won't be able to pick bugs folded in with PRs.

    Parameters
    ----------
    search_type : {'loose', 'tight', 'major'}
        If loose, searches for the full spectrum of terms that probably flag
        a bug. If tight, seaches strictly for "bug" & variants. If major,
        searches for "major bug", "severe bug", "significant bug".
    message_headline : str
        The topic of a commit.
    message : str
        The body of a commit.

    Examples
    --------
    >>> message = 'nope'
    >>> header = 'noooooooope'
    >>> is_commit_bug('loose', header, message)
    False
    >>> message1 = 'mending the code'
    >>> message2 = 'Changing the code'  # this fails
    >>> message3 = 'Major bug'
    >>> header1 = 'Commit'  # this fails
    >>> header2 = 'Bugs all over the place'
    >>> is_commit_bug('loose', message1, header1)
    True
    >>> is_commit_bug('tight', message1, header1)
    False
    >>> is_commit_bug('loose', message2, header1)
    False
    >>> is_commit_bug('tight', message2, header1)
    False
    >>> is_commit_bug('loose', message2, header2)
    True
    >>> is_commit_bug('tight', message2, header2)
    True
    >>> is_commit_bug('loose', message1, header2)
    True
    >>> is_commit_bug('major', message1, header2)
    False
    >>> is_commit_bug('major', message3, header2)
    True
    """
    assert search_type in ('loose', 'tight', 'major')
    # regex syntax reminder: ^ start of line, $ end of line, \W not word char
    # s? s zero or one time
    bug1 = r'(^|\W)[Bb]ugs?($|\W)'
    bug2 = r'(^|\W)[Bb]uggy($|\W)'
    mend = r'(^|\W)[Mm]end(ing)?s?($|\W)'
    broken = r'(^|\W)[Bb]roken($|\W)'
    forgot = r'(^|\W)[Ff]orgot($|\W)'
    worksright = r'(^|\W)works? right($|\W)'
    workscorrectly = r'(^|\W)works? correctly($|\W)'
    dealwith = r'(^|\W)[Dd]eals? with($|\W)'
    typo = r'(^|\W)[Tt]ypo($|\W)'
    wrong = r'(^|\W)[Ww]rong($|\W)'
    fix = r'(^|\W)[Ff]ix(es)?($|\W)'
    allposs = (
        bug1 + '|' + bug2 + '|' + mend + '|' + broken + '|'
        + forgot + '|' + worksright + '|' + workscorrectly + '|'
        + dealwith + '|' + typo + '|' + wrong + '|' + fix
    )
    tight = bug1 + '|' + bug2
    major = (
        r'(^|\W)[Mm]ajor bugs?($|\W)|'
        + r'(^|\W)[Ss]ignificant bugs?($|\W)|'
        + r'(^|\W)[Ss]evere bugs?($|\W)'
    )
    found = False
    if search_type == 'loose':
        search_terms = allposs
    elif search_type == 'tight':
        search_terms = tight
    elif search_type == 'major':
        search_terms = major
    for mess in (message_headline, message):
        found = bool(re.search(search_terms, mess)) or found
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


def build_commit_and_bug_timelines(commits, search_type):
    """
    Creates timelines of when commits were made and bugs were found.
    Also tracks the authors committing to the repo, and if COUNT_ADDITIONS,
    the number of added lines associated with each commit.

    Parameters
    ----------
    commits : list of dicts
        the dicts output from the graphql searches on Github.
    search_type : ('loose', 'tight', 'major')
        Search constraints for bug terms (see is_commit_bug)
    """
    dtimes = []
    times_bugs_fixed = []
    last_dtime = None
    last_bug_fix = None
    authors = set()
    additions = []
    # firsttime = first_commit_dtime(commits, False, override_next_page=True)
    for auth, dtime, head, mess, adds in yield_commits_data(commits):
        authors.add(auth)
        isbug = is_commit_bug(search_type, head, mess)
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


def calc_averages_for_intervals(commits_in_order, bug_fraction_in_order):
    """Calc the mean value of bug_fraction for the intervals
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 20, 28, 42, 90, 1000000]
    (selected so once commits>4, each interval contains ~50 repos)
    """
    bin_intervals = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 20, 28, 42, 90, 1000000]
    )
    bin_vals = []
    bin_count = []
    bbase_index = 0
    for btop in bin_intervals[1:]:
        btop_index = np.searchsorted(commits_in_order, btop, 'right')
        bin_vals.append(np.mean(
            bug_fraction_in_order[bbase_index:btop_index]
        ))
        bin_count.append(btop_index - bbase_index)
        bbase_index = btop_index
    return np.array(bin_vals), np.array(bin_count)



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
        # But, seems like we're seeing bursts of bug finding, all with similar-
        # ish rising gradients, interspersed with periods of code growth.
        # The steps don't notably change gradient with amount of material added
        # (though hard to judge). This would imply there are still plenty of
        # bugs to find the whole time.

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


# def fit_curvature(x, y, centering):
#     """
#     Fits the equation c0 + c1*x + c2*x**2 to the data y, then calculates the
#     radius of curvature as (1 + (dy/dx)**2)**3/2 / d^2y/dx^2, at x=centering.
#
#     Positive is concave-up, i.e., bug find rate accelerates through time.
#
#     This works, but not well tailored to the application yet.
#     """
#     try:
#         c = np.polyfit(x, y, 2)
#     except np.linalg.LinAlgError:
#         return None
#     dybydx = np.poly1d([2. * c[0], c[1]])
#     d2ybydx2 = np.poly1d([2. * c[0]])
#     dybydx2atpt = dybydx(centering) ** 2
#     numerator = (1 + dybydx2atpt) ** 1.5
#     return numerator / d2ybydx2(centering)


def area_from_curve_to_abscissa(abscissa_y, points_on_line_x, points_on_line_y):
    """
    Calculates the area between a line defined parallel to x (the abscissa) and
    points defining the line in (x, y) space. Can be positive (area below the
    line) or negative (area above the line).

    Assumes points are in x-order, with no bad data.
    """
    assert np.all(np.diff(points_on_line_x) >= 0.)  # ordered correctly
    dists_to_abscissa = points_on_line_y - abscissa_y
    avg_dist_each_trapezium = (dists_to_abscissa[:-1]
                               + dists_to_abscissa[1:]) / 2.
    spacing_each_trapezium = points_on_line_x[1:] - points_on_line_x[:-1]
    trapezium_areas = avg_dist_each_trapezium * spacing_each_trapezium
    total_area = np.sum(trapezium_areas)
    return total_area


def spatial_percentile(percentile, points_on_line_x, points_on_line_y):
    """
    Calculate a percentile for a dataset where the data are weighted by their
    spacing. Percentile specified as a fraction.

    Examples
    --------

    """
    assert 0. <= percentile <= 1.
    # First, uniquely associate each value with its spacing. This is just a
    # trapezium rule approach:
    spacing_each_trapezium = points_on_line_x[1:] - points_on_line_x[:-1]
    val_at_each_trapezium = (points_on_line_y[:-1] + points_on_line_y[1:]) / 2.
    area_each_trapezium = spacing_each_trapezium * val_at_each_trapezium
    # order the data by y:
    order_for_vals = np.argsort(val_at_each_trapezium)
    accumulating_areas = np.cumsum(area_each_trapezium[order_for_vals])
    total_area = accumulating_areas[-1]
    # find the percentile in the accumulating areas:
    pc_position = np.searchsorted(accumulating_areas,
                                      percentile * total_area)
    pc_value = val_at_each_trapezium[order_for_vals][pc_position]
    return pc_value


def cloc_repo(repo_nameowner):
    """
    Takes a nameowner for a Github repo and returns a dict of scientific
    programming languages used and the number of lines of each.
    Requires the command line tool cloc.
    """
    assert type(repo_nameowner) is str
    dict_language_to_codelines = {}
    print('Running cloc on', repo_nameowner)
    bashscript = 'git clone --depth 1 https://github.com/'
    bashscript += repo_nameowner + '.git temp-linecount-repo &&\n'
    bashscript += 'cloc --sql 1 temp-linecount-repo | sqlite3 repo_cloc.db &&\n'
    bashscript += 'rm -rf temp-linecount-repo\n'
    os.system(bashscript)
    # con = sqlite3.connect('repo_cloc.db')
    # try:
    #     out = pandas.read_sql('SELECT * FROM t', con)
    # except (AttributeError, con.OperationalError, con.DatabaseError):
    #     print('No table for', repo_nameowner)
    #     os.system('rm repo_cloc.db')
    #     return {}
    # ^^ this creates unhandlable errors if no table, so use sqlalchemy direct:
    engine = sqlalchemy.create_engine('sqlite:///repo_cloc.db')
    if engine.has_table('t'):
        print('Loading the table...')
        out = pandas.read_sql('SELECT * FROM t', engine)
    else:
        print('No table for', repo_nameowner)
        #os.system('rm repo_cloc.db')
        os.remove('repo_cloc.db')
        # some errors leave this in place so:
        os.system('rm -rf temp-linecount-repo')
        return {}
    #
    #os.system('rm repo_cloc.db')
    os.remove('repo_cloc.db')
    total_lines_of_code = out['nCode'].sum()
    for lang in LANGUAGES_TO_TEST_FOR:
        lines_in_lang = out['nCode'][out['Language'] == lang]
        dict_language_to_codelines[lang] = int(lines_in_lang.sum())
    print('***')
    return dict_language_to_codelines


def repo_lengths_to_file(repo_nameowners):
    """
    Adds to a json file called repo_lengths.json in top level of the dir
    structure that records the length of repos called nameowner.

    Also returns the dict, so doubles as a loader.
    """
    if os.path.exists('repo_lengths.json'):
        with open('repo_lengths.json', 'r') as infile:
            length_dict = json.load(infile)
    else:
        length_dict = {}
    for repo in repo_nameowners:
        if repo not in length_dict.keys():
            lines = cloc_repo(repo)
            length_dict[repo] = lines
    with open('repo_lengths.json', 'w') as outfile:
        json.dump(length_dict, outfile)
    return_dict = {repo:entry for (repo, entry) in length_dict.items()
                   if repo in repo_nameowners}
    return return_dict


if __name__ == "__main__":
    topic = 'https://doi.org'  # 'landlab', 'terrainbento', 'physics', 'chemistry', 'https://doi.org', 'biolog'
    # the search for Landlab isn't pulling landlab/landlab as a long repo!? Check
    search_type = 'loose'  # for how to pick bugs ('loose', 'tight', 'major')
    search_fresh = False
    continue_old_saves = True
    # ^If true, script begins by a fresh call to the API and then a save
    # If false, proceeds with saved data only
    if search_fresh:
        approx_desired_repos = 500
        approx_max_commits = 2000
        if COUNT_ADDITIONS:
            # not yet quite stable
            print('Searching for commit lengths, this might be slow...')
            get_data_limit = 8  # these terms matter for stability
            long_repo = 50
        else:
            get_data_limit = 20
            long_repo = 100

        # leave this section alone:
        pages = approx_desired_repos // get_data_limit + 1
        max_iters_for_commits = approx_max_commits // HISTORY_PAGE
        cursor = None

    print('Searching on ' + topic)
    if search_fresh:
        print('Calling the GitHub API...')
    else:
        print('proceeding with saved data...')
    bug_find_rate = []  # i.e., per bugs per commit
    total_authors = []
    total_bugs_per_repo = []
    commit_rate_median_per_repo = []
    commit_rate_mean_per_repo = []
    bug_rate_median_per_repo = []
    bug_rate_mean_per_repo = []
    # all_repos = []  # will store [num_commits, nameowner]
    short_repos = []  # will store [num_commits, nameowner, name, owner]
    long_repos = []  # will store [num_commits, nameowner, name, owner]
    coveralls_count = []
    total_commits_from_API = []
    if COUNT_ADDITIONS:
        total_lines_from_API = []
        bug_find_rate_per_line = []  # i.e., bugs per line
        bugs_per_line_infinite_count = 0  # used for tracking bug finds, no lines added

    # do the API call fresh if needed:
    if search_fresh:
        get_process_save_data_all_repos(
            pages, get_data_limit, topic, long_repo, cursor, HEADER,
            continue_run=continue_old_saves
        )
        print('Now interrogating long repos. This might be slow...')
        get_process_save_commit_data_long_repos(topic, HEADER,
                                                max_iters=max_iters_for_commits)

    # now load and proceed:
    for enum, (
        rep_data, nameowner, name, owner, creation_date,
        last_push_date, commit_page_data, has_next_page,
        commits, total_commits, languages, readme_text
    ) in enumerate(load_processed_data_all_repos(topic, 'short')):
        badges = look_for_badges(readme_text)

        times_bugs_fixed, dtimes, authors, additions = \
            build_commit_and_bug_timelines(commits, search_type)

        short_repos.append([total_commits, nameowner, name, owner])
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
        if COUNT_ADDITIONS:
            total_lines_added = np.sum(additions)
            total_lines_from_API.append(total_lines_added)
            try:
                bug_find_rate_per_line.append(
                    len(times_bugs_fixed) / total_lines_added
                )
            except ZeroDivisionError:
                bug_find_rate_per_line.append(float('nan'))
                bugs_per_line_infinite_count += 1

        try:
            bug_from_start_time, from_start_time = \
                build_times_from_first_commit(times_bugs_fixed, dtimes)
        except TypeError:  # no commits present
            print(nameowner)

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

    # print('*****')
    # for repo in sorted(long_repos)[::-1]:
    #     print(repo)

    print('*****')
    num_short_repos = len(commit_rate_mean_per_repo)
    short_count = len(coveralls_count)
    print('Of ' + str(num_short_repos) + ' short repositories, '
          + str(short_count) + ' use coveralls')
    if short_count > 0:
        print("They are:")
        for ln in coveralls_count:
            print(ln)

    # print('***')
    # print('Found ' + str(len(long_repos)) + ' long repos.')
    # # input('Proceed? [Enter]')

    # load the save long repo data
    long_repo_commit_dict = load_processed_commit_data_long_repos(topic)

    for enum_long, (
        rep_data, nameowner, name, owner, creation_date,
        last_push_date, commit_page_data, has_next_page,
        commits, total_commits, languages, readme_text
    ) in enumerate(load_processed_data_all_repos(topic, 'long')):
        badges = look_for_badges(readme_text)
        commits = long_repo_commit_dict[nameowner]
        print('Successfully loaded ' + str(len(commits)) + ' commits')
        # try:
        times_bugs_fixed, dtimes, authors, additions = \
            build_commit_and_bug_timelines(commits, search_type)
        # except TypeError:
        #     # bad data
        #     continue
        long_repos.append([total_commits, nameowner, name, owner])
        total_authors.append(len(authors))
        total_bugs_per_repo.append(len(times_bugs_fixed))
        total_commits_from_API.append(total_commits)
        try:
            bug_find_rate.append(len(times_bugs_fixed) / len(dtimes))
        except ZeroDivisionError:
            bug_find_rate.append(0.)
        if COUNT_ADDITIONS:
            total_lines_added = np.sum(additions)
            total_lines_from_API.append(total_lines_added)
            try:
                bug_find_rate_per_line.append(
                    len(times_bugs_fixed) / total_lines_added
                )
            except ZeroDivisionError:
                bug_find_rate_per_line.append(float('nan'))
                bugs_per_line_infinite_count += 1

        bug_from_start_time, from_start_time = \
            build_times_from_first_commit(times_bugs_fixed, dtimes)

        if 'coveralls' in badges:
            coveralls_count.append([enum_long + num_short_repos, owner, name])
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
    long_repos = total_repos - num_short_repos
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
    # also a moving average excluding the rate = 0 cases
    # (=> is it just them causing the trend?)
    # plotting will show the trend is still there without zeros
    rate_not_zero = np.logical_not(np.isclose(bug_find_rate_ordered, 0.))
    bug_find_rate_moving_avg_no_zeros = moving_average(
        bug_find_rate_ordered[rate_not_zero], n=20
    )
    commits_more_than_1 = total_commits_IN_order > 1
    bug_find_rate_moving_avg_not1commit = moving_average(
        bug_find_rate_ordered[commits_more_than_1], n=20
    )

    plot(total_commits_from_API, bug_find_rate, 'x')
    plot(total_commits_IN_order[:-19], bug_find_rate_moving_avg, '-')
    plot(total_commits_IN_order[rate_not_zero][:-19],
         bug_find_rate_moving_avg_no_zeros, '-')
    plot(total_commits_IN_order[commits_more_than_1][:-19],
         bug_find_rate_moving_avg_not1commit, '-')
    # ^^...this makes no difference really, as it only changes one point
    plot(total_commits_from_API_array[cov_indices],
         bug_find_rate_array[cov_indices], 'kx')  # coveralls ones in black
    xlabel('Total number of commits')
    ylabel('Fraction of all commits finding bugs')

    figure('bug rate per line vs cumulative lines')
    plot(total_lines_from_API, bug_find_rate_per_line, 'x')

    # calculate the total number of bugs missing from the shorter projects
    # take the final value on the moving avg as the idealised perfect find rate
    # could revisit this assumption
    theoretical_find_rate = bug_find_rate_moving_avg[-1]
    all_missing_bugs = area_from_curve_to_abscissa(
        theoretical_find_rate, total_commits_IN_order[:-19],
        bug_find_rate_moving_avg
    )
    assert all_missing_bugs < 0.  # check!
    all_found_bugs = area_from_curve_to_abscissa(
        0., total_commits_IN_order[:-19], bug_find_rate_moving_avg
    )
    assert all_found_bugs > 0.
    bugs_not_found_in_typical_repo = -all_missing_bugs / all_found_bugs
    # print('Bug fraction not found in a typical repo:',
    #       bugs_not_found_in_typical_repo)
    # does changing the centering of the moving avg meaningfully change this?
    # YES, starting from zero (i.e., [:-19], not [10:-9]) increases the
    # fraction from ~23% to ~33%.
    # Which should we prefer? If offset, maybe we are missing the left hand
    # rectangle in our trapezium rule?2

    all_missing_bugs_no_zeros = area_from_curve_to_abscissa(
        theoretical_find_rate, total_commits_IN_order[rate_not_zero][10:-9],
        bug_find_rate_moving_avg_no_zeros
    )
    all_found_bugs_no_zeros = area_from_curve_to_abscissa(
        0., total_commits_IN_order[rate_not_zero][10:-9],
        bug_find_rate_moving_avg_no_zeros
    )
    bugs_not_found_in_typical_repo_no_zeros = (
        -all_missing_bugs_no_zeros / all_found_bugs_no_zeros
    )
    print('Bug fraction not found in a typical repo, discounting zero bug repos:',
          bugs_not_found_in_typical_repo_no_zeros)

    # We could picture this as a mixing line, between two idealised end members:
    # one where we have mature code that finds bugs at the idealised 20% rate,
    # and a completely immature one, where tons of bugs could be present but
    # none are found. If we do this, then these fractions ARE the fractions of
    # those two end members (i.e., 33% immature, 67% mature - note this is
    # pleasingly close to the fraction of repos that find no bugs at all,
    # despite not making this assumption).
    # This number is VERY sensitive to the final steady state bug find rate;
    # dropping it from 0.2 to 0.175 more or less halves the typical bug rate
    # to 16% from 33%.
    # This could be addressed semi-formally by truncating the final ?30 pts
    # off the data set, doing the differencing only before that, using the
    # distribution in the 30 to get a standard error on the mean, then using the
    # bounds to assess the possible range.
    # (This should probably be a moving mean, not median!)
    # Getting the median fraction might be more meaningful - but needs careful
    # thinking about how to get this out stably, i.e., not just the halfway
    # point, which will suffer lots of noise.
    # In reality though, we don't want an end member, because codes may be
    # added to github in a variety of states of maturity. This probably doesn't
    # affect the fraction of bugs missed in typical code though, as we can
    # broadly equate the full set of code that gets to github with the
    # condition of the legacy code base. i.e., we're effectively sampling the
    # same thing.

    median_bug_find_rate_using_moving_avg = spatial_percentile(
        0.5, total_commits_IN_order[:-19], bug_find_rate_moving_avg
    )
    # so the ratio becomes...
    median_bugs_not_found_in_typical_repo = (
        theoretical_find_rate - median_bug_find_rate_using_moving_avg
    ) / median_bug_find_rate_using_moving_avg
    # ...which is 0.26 with tfr=0.20 and 0.097 with tfr=0.175

    # So let's constrain tfr as we said above:
    last_set_of_bug_find_rates = bug_find_rate_ordered[-20:]
    theoretical_find_rate_mean = np.mean(last_set_of_bug_find_rates)
    theoretical_find_rate_std = np.std(last_set_of_bug_find_rates)
    theoretical_find_rate_95_pciles = (
        1.96 * theoretical_find_rate_std / np.sqrt(20)
    )
    # i.e., mean find rate is
    min_mean_max_tfrs = (
        theoretical_find_rate_mean - theoretical_find_rate_95_pciles,
        theoretical_find_rate_mean,
        theoretical_find_rate_mean + theoretical_find_rate_95_pciles
    )
    min_mean_max_bugs_not_found_in_typical_repo = []
    for tfr in min_mean_max_tfrs:
        all_missing_bugs = area_from_curve_to_abscissa(
            tfr, total_commits_IN_order[:-19],
            bug_find_rate_moving_avg
        )
        bugs_not_found_in_typical_repo = -all_missing_bugs / all_found_bugs
        min_mean_max_bugs_not_found_in_typical_repo.append(
            bugs_not_found_in_typical_repo
        )
    # this produces mean fractions 0.33, down to 0.12 (min) and up to 0.55 (max)
    # We can do the same thing with the median, since 2sigma should bracket 95%
    # of the data:
    min_median_max_bug_fraction_in_typical_repo = []
    for pc in (0.025, 0.5, 0.975):
        pc_bugs_not_found_in_typical_repo = spatial_percentile(
            pc, total_commits_IN_order[:-19], bug_find_rate_moving_avg
        )
        min_median_max_bug_fraction_in_typical_repo.append(
            pc_bugs_not_found_in_typical_repo
        )
    # (Do we need to do this, or can we just get medians from the avg val
    # uncertainty?)

    # So, ~0.33 of the bugs remain in a typical script... The logic so far
    # says that tfr is the bug rate per change found in well-run code, i.e.,
    # 20%. So if we know the mean length of a commit, we can estimate the
    # number of bugs per line.

    # We can use cloc (brew install cloc) to count lines of code, per
    # https://stackoverflow.com/questions/26881441/can-you-get-the-number-of-lines-of-code-from-a-github-repository
    # See the cloc_repo func above
    # don't run unless you want to wait a bit...
    # for num_commits, nameowner in all_repos:
    #     lines_of_code = cloc_repo(nameowner)

    # If we look only for major bugs, we find only two in the whole of the
    # 200 repos for physics (1500 commits max).

    # What you get with a tight definition of bug is intriguing.
    # Based on the 20 longest commit sequences, the mean of "steady" bug
    # finding is 0.045 (0.021-> 0.069). Chemistry is 0.037 (0.023->0.052)
    # i.e., by the time you add 20 commits, you would expect one bug

    # There are some repos that *are not* code (e.g. learning resources).
    # We could thin these out by checking for language = Python, C, Javascript,
    # etc.

    # What is the commit number distribution of the zero bug code?
    figure('Commits in zero bug repos')
    zero_bug_repos = np.isclose(bug_find_rate_ordered, 0.)
    commits_of_zero_bug_repos = total_commits_IN_order[zero_bug_repos]
    hist(commits_of_zero_bug_repos, bins='auto', range=(0, 1000))
    xlabel('Number of commits')
    ylabel('Number of repos')

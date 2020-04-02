import re
import io
from matplotlib.pyplot import plot, figure

# convert_datetime = io.convert_datetime

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
        pushed_datetime = convert_datetime(data['pushedDate'])
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

# we should also be tracking BUG ISSUES, as can't see PR merges from off master

dtimes = []
times_bugs_fixed = []
time_to_bug_fix = []
commit_rate = []
bug_fix_rate = []
last_dtime = None
last_bug_fix = None
authors = set()
firsttime = first_commit_dtime(commits, False, override_next_page=True)
for auth, dtime, head, mess in yield_commits_data(commits):
    authors.add(auth)
    isbug = is_commit_bug(head, mess)
    print(isbug)
    if dtime is not None:
        dtimes.append(dtime)
        if last_dtime is None:
            commit_rate.append(None)
        else:
            commit_rate.append(1./(dtime - last_dtime).seconds)
        if isbug:
            try:
                bug_fix_rate.append(1./(dtime - last_bug_fix).seconds)
            except TypeError:  # None
                bug_fix_rate.append(None)
            times_bugs_fixed.append(dtime)
            try:
                time_to_bug_fix.append((dtime - firsttime).seconds)
            except TypeError:
                time_to_bug_fix.append(None)
            last_bug_fix = dtime
        last_dtime = dtime

i = list(range(len(dtimes)))
plot(dtimes, i)

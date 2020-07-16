import requests, json, re, os, pandas, sqlite3, time, sqlalchemy
import numpy as np
from matplotlib.pyplot import plot, figure, show, xlabel, ylabel, xlim, ylim, bar, hist
from datetime import datetime
from bug_model.header.header import HEADER
from requests.auth import HTTPDigestAuth
from copy import copy
from sqlalchemy.exc import OperationalError, DatabaseError
from bug_model.utils import moving_average

with open('repo_lengths.json') as f:
    language_lengths = json.load(f)

repo_lengths = []
for r in language_lengths.values():
    repo_lengths.append(np.sum(list(r.values())))

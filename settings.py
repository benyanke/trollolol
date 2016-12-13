# Django settings for celerytest project.

import os
import sys
# from django.conf.global_settings import TEMPLATE_CONTEXT_PROCESSORS
# os.environ["CELERY_LOADER"] = "django"

PROJECT_ROOT = os.path.dirname(__file__)

DEBUG = True
TEMPLATE_DEBUG = DEBUG

ADMINS = (
    # ('Your Name', 'your_email@example.com'),
)

# Troll classifier files
TROLL_TRAIN = PROJECT_ROOT + '/insult_datasets/train.csv'
TROLL_TEST = PROJECT_ROOT + '/insult_datasets/test.csv'
TROLL_WORDS = PROJECT_ROOT + '/reddit_comments/feature_words.txt'

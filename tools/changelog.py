#!/usr/bin/python
# from https://github.com/macro6461/changelog-python

from datetime import date
from os.path import exists
today=date.today()
version="1.2.3"
item = "## [{}] - {}".format(version, today)
# use below to correspond with your tagged version
# version=subprocess.check_output(["git", "describe", "--long"]).strip()
changelog_start="""# Changelog
All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [Unreleased]
{}
### Added
- ADD CHANGE HERE!
""".format(item)

def read_changelog():
    changelog = open('CHANGELOG.md', 'r')
    changelog_items = changelog.readlines()
    return changelog_items

def write_changelog_lines(changelog_items):
    with open('CHANGELOG.md', 'w') as changelog:
        changelog.writelines(changelog_items)

def changelog_helper(item):
    changelog_items=read_changelog()
    does_exist=False 
    for line in changelog_items:
        if item.strip() in line.strip():
            does_exist=True
            break
    
    return does_exist

def new_changelog_item():
    is_found=changelog_helper(item)
    if (is_found):
        print("Changelog item already exists for \n    {}".format(item))
    else:
        index = -1
        changelog_items=read_changelog()
        for line in changelog_items:
            index+=1  
            if "## [Unreleased]" in line:
                changelog_items[index] = "## [Unreleased]\n{}\n### Added\n- ADD CHANGE HERE!\n".format(item)
                write_changelog_lines(changelog_items)
                break   

def new_changelog():
    changelog = open("CHANGELOG.md", "w")
    changelog.write(changelog_start)
    changelog.close()

def init():
    if exists('CHANGELOG.md'):
        new_changelog_item()
    else:
        new_changelog()

init()
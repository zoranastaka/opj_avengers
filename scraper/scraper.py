"""
Create create_php_dict will return a list of repo_link: list_of_files pairs.
"""

# HTML parsing libraries:
from bs4 import BeautifulSoup
import urllib.request as urllib2
import json


def get_repo_url(repo):
    """
    Find link to the repository from the bs4.element.Tag containing repository information.

    :param repo: bs4.element.Tag containing repository information
    :return: Link to the repository (str)
    """
    return repo.find(
        'div', {"class": 'mt-n1'}
    ).find(
        'div', {"class": "f4 text-normal"}
    ).find('a').get('data-hydro-click').split('"url"')[1].split('"')[1]


def get_repo_files(repo_url, limit=None, dir_limit=None, extension='.php'):
    """
    Make a list of all files with a certain extension in a repository.

    :param repo_url:
    :param limit: If set, return maximum of limit files in the list. (default: None)
    :param dir_limit: How many files to get per directory.
    :return: List of file urls from the repository.
    """

    file_and_dir_class = "js-navigation-open Link--primary"

    repo = urllib2.urlopen(repo_url)
    soup = BeautifulSoup(repo, 'html.parser')

    files_and_directories = [object.get('href') for object in soup.find_all('a', {"class": file_and_dir_class})]
    files = [file for file in files_and_directories if 'blob' in file]
    dirs = [file for file in files_and_directories if 'blob' not in file]

    useful_files = [file for file in files if file.endswith(extension)]
    processed_files = [file.replace('blob', 'raw') for file in useful_files]

    if limit is not None:
        if len(processed_files) > limit:
            return processed_files[0:limit].copy()
        else:
            for directory in dirs:
                curr_dir_url = 'https://github.com' + directory
                files_from_dir = get_repo_files(
                    repo_url=curr_dir_url, limit=dir_limit, dir_limit=dir_limit, extension=extension
                )
                processed_files = processed_files + files_from_dir
                if len(processed_files) > limit:
                    break
        if len(processed_files) > limit:
            return processed_files[0:limit].copy()
        else:
            return processed_files.copy()
    else:
        for directory in dirs:
            curr_dir_url = 'https://github.com' + directory
            files_from_dir = get_repo_files(
                repo_url=curr_dir_url, limit=dir_limit, dir_limit=dir_limit, extension=extension
            )
            processed_files = processed_files + files_from_dir
        return processed_files


def create_php_dict(url, limit=5000, max_files_per_repo=50, max_files_per_dir=10):
    """
    Generate a set of repo-file objects. Search page should be limited to repositories only!
    Sort and keywords are optional.

    :param url: Base url to github search page
    :param limit: Number of files to return. Files are returned as
    :param max_files_per_repo: How many files per repo to return? Point is to have a more diverse set of code blocks
    :param max_files_per_dir: How many files per directory to return.
    :return: list of dicts. Each dict contains repo url and list of files from said repo
    """

    total_added_files = 0
    php_dict = {}
    curr_url = url

    while total_added_files < limit:
        # Get repo list from the current page:

        curr_page = urllib2.urlopen(curr_url)
        soup = BeautifulSoup(curr_page, 'html.parser')
        curr_page_repo_objects = soup.find_all("li", {"class": "repo-list-item"})
        curr_page_repos = [get_repo_url(repo) for repo in curr_page_repo_objects]

        # Collect php files from the repo
        for curr_repo in curr_page_repos:
            curr_repo_files = get_repo_files(repo_url=curr_repo, limit=max_files_per_repo, dir_limit=max_files_per_dir)
            total_added_files += len(curr_repo_files)
            php_dict[curr_repo] = curr_repo_files.copy()
            print("Finished processing {}. Added total of {} files".format(curr_repo, len(curr_repo_files)))
            if total_added_files > limit:
                break

        # Go to next page
        print("Went to the next page")
        curr_url = 'https://github.com' + soup.find('a', {"class": "next_page"}).get('href')

    return php_dict


def store_dict(data, filename='data.json'):
    """
    Writes dictionary into a json file.

    :param data: Dictionary created using create_php_dict() function.
    :param filename: Name of the .json file in which to write data
    :return: Nothing
    """
    with open(filename, 'w') as json_file:
        json.dump(data, json_file)


def read_json(path):
    """
    Reads data from a json file into a dictionary.

    :param path: Path to the file.
    :return: Dictionary
    """

    with open(path) as json_file:
        data = json.load(json_file)

    return data


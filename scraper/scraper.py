"""
Create create_php_dict will return a list of repo_link: list_of_files pairs.
"""

# HTML parsing libraries:
from bs4 import BeautifulSoup
import urllib.request as urllib2


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
        return processed_files.copy()


def create_php_dict(url, limit=5000, max_files_per_repo=50):
    """
    Generate a set of repo-file objects.

    :param url: Base url to github search page
    :param limit: Number of files to return. Files are returned as
    :param max_files_per_repo: How many files per repo to return? Point is to have a more diverse set of code blocks
    :return: list of dicts. Each dict contains repo url and list of files from said repo
    """

    total_added_files = 0
    php_dict = {}

    while total_added_files < limit:
        # Get repo list from the current page:

        curr_page = urllib2.urlopen(url)
        soup = BeautifulSoup(curr_page, 'html.parser')
        curr_page_repo_objects = soup.find_all("li", {"class": "repo-list-item"})
        curr_page_repos = [get_repo_url(repo) for repo in curr_page_repo_objects]

        # Collect php files from the repo
        for curr_repo in curr_page_repos:
            curr_repo_files = get_repo_files(repo_url=curr_repo, limit=max_files_per_repo)
            total_added_files += len(curr_repo_files)
            php_dict[curr_repo] = curr_repo_files.copy()


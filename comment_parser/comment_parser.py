import urllib
import re
import codecs
import pandas as pd

from typing import List
from scraper.scraper import read_json


class Comment:
    def __init__(self, text, useful_text):
        self.text = text
        self.useful_text = useful_text

    def __repr__(self):
        temp_dict = {
            'text': self.text,
            'useful_text': self.useful_text
        }
        return str(temp_dict)

    def __str__(self):
        temp_dict = {
            'text': self.text,
            'useful_text': self.useful_text
        }
        return str(temp_dict)


class Code_comment:
    def __init__(self, repo_desc, source_desc, code, comment: Comment):
        self.pair_id = None
        self.repo_desc = repo_desc
        self.source_desc = source_desc
        self.code = code
        self.comment: Comment = comment

    def __repr__(self):
        temp_dict = {
            'pair_id': self.pair_id,
            'repo_desc': self.repo_desc,
            'source_desc': self.source_desc,
            'code': self.code,
            'comment': self.comment
        }
        return str(temp_dict)

    def __str__(self):
        temp_dict = {
            'pair_id': self.pair_id,
            'repo_desc': self.repo_desc,
            'source_desc': self.source_desc,
            'code': self.code,
            'comment': self.comment
        }
        return str(temp_dict)


code_comments_list: List[Code_comment] = []


def extract_comments(php_path='php_files.json', verbose=False):
    """
    Extracts comment-code pairs from a json file where keys are repository names and values are lists of relative file paths.

    :param php_path: Path to the json file.
    :param verbose: If True prints progress to the command line more often. Default: False
    :return:
    """
    php_dict = read_json(php_path)

    counter = 0
    for key, values in php_dict.items():
        for value in values:
            try:
                extract_comments_from_file(key, value, verbose=verbose)
            except urllib.error.HTTPError:
                print("Skipping {}.".format(value))
                continue
            except UnicodeDecodeError:
                print("Skipping {} due to UnicodeDecodeError.".format(value))

        counter += 1
        if counter % 5 == 0:
            print('Processed {} repositories.'.format(counter))

    export_data()


def extract_code(lines, code_line_no, comment_text, useful_text, repo_desc, source_desc):
    code = ''
    if ('function ' in lines[code_line_no] or 'class ' in lines[code_line_no]) and '=' not in lines[code_line_no]:
        code += lines[code_line_no]
        comment = Comment(comment_text, useful_text)

        # find all the code
        num_brackets = 0

        code_line_no += 1
        while lines[code_line_no] == '\n':
            code_line_no += 1
        if '{' in lines[code_line_no]:
            num_brackets += 1
            code += lines[code_line_no]
            code_line_no += 1

        num_left_bracket = 0
        num_right_bracket = 0
        while num_brackets != 0:
            try:
                if '{' in lines[code_line_no] and '}' in lines[code_line_no]:
                    pass
                elif '{' in lines[code_line_no]:
                    num_brackets += 1
                    num_left_bracket += 1
                elif '}' in lines[code_line_no]:
                    num_brackets -= 1
                    num_right_bracket += 1
                code += lines[code_line_no]
                code_line_no += 1
            except:
                pass

        code_comment = Code_comment(repo_desc, 'https://github.com' + source_desc, code, comment)
        code_comments_list.append(code_comment)


def extract_comments_from_file(repo_desc, source_desc, verbose=False):
    """
    Extracts comment-code pairs from a single file

    :param repo_desc: Path to the repository containing the file.
    :param source_desc: Relative path to the file.
    :param verbose: If True prints progress to the command line more often. Default: False
    :return:
    """
    comment_text = ''
    useful_comment_text = ''

    regex_pattern_start_multiline = r'^([\s]*)\/\*'
    regex_pattern_multiline = r'^([\s]*)\*'
    regex_pattern_end_multiline = r'^([\s]*)\*\/'
    regex_pattern_single = r'^([\s]*)(\#|\/\/)'

    url_php_file = 'https://github.com' + source_desc
    if verbose:
        print(url_php_file)
    response = urllib.request.urlopen(url_php_file)

    lines = response.readlines()

    lines = [x.decode('utf-8') for x in lines]

    line_no = 0
    comment_no_in_file = 0
    while line_no < len(lines):
        decoded_line = lines[line_no]

        # multiline comment starting with /*, ending with */, spanning multiple lines starting with *
        if re.match(regex_pattern_start_multiline, decoded_line) is not None:
            comment_text += decoded_line
            line_no += 1
            while line_no < len(lines) and re.match(regex_pattern_multiline, lines[line_no]) is not None:
                comment_text += lines[line_no]
                if not re.match(regex_pattern_end_multiline, lines[line_no]):
                    useful_comment_text += lines[line_no].split('*')[1]
                line_no += 1

            while line_no < len(lines) and lines[line_no] == '\n':
                line_no += 1

            if line_no < len(lines):
                extract_code(lines, line_no, comment_text, useful_comment_text, repo_desc, source_desc)
            comment_text = ''
            useful_comment_text = ''

        # single line comment starting with # or //
        elif re.match(regex_pattern_single, decoded_line):
            while line_no < len(lines) and re.match(regex_pattern_single, lines[line_no]):
                comment_text += lines[line_no]
                if len(lines[line_no].split('#')) == 1:
                    useful_comment_text += lines[line_no].split('//')[1]
                else:
                    useful_comment_text += lines[line_no].split('#')[1]
                line_no += 1

            while line_no < len(lines) and lines[line_no] == '\n':
                line_no += 1

            if line_no < len(lines):
                extract_code(lines, line_no, comment_text, useful_comment_text, repo_desc, source_desc)

            comment_text = ''
            useful_comment_text = ''

        else:
            line_no += 1


def prepare_data_for_export():
    add_comments_id()
    encode_new_line_and_tab_in_comments()


def add_comments_id():
    for i in range(len(code_comments_list)):
        repo_part = code_comments_list[i].repo_desc.split('/')[-2:]
        source_part = code_comments_list[i].source_desc.split('/')[-1:]
        try:
            code_comments_list[i].pair_id = repo_part[0] + '_' + repo_part[1] + '_' + source_part[0][:-4] + '_' + str(i + 1)
            # print("Success!")
            # print(code_comments_list[i].repo_desc)
        except IndexError as err:
            print("Error!")
            print(code_comments_list[i].repo_desc)
            raise


def export_pairs_to_txt_files():
    for pair in code_comments_list:
        file_code_comment_pair = codecs.open('data/pairs/' + pair.pair_id + '.txt', 'w+', 'utf-8')
        # print("Wrote to {}.".format(pair.pair_id+'.txt'))
        text_to_write = pair.comment.text + '\n' + pair.code
        file_code_comment_pair.write(text_to_write)
        file_code_comment_pair.close()


def encode_new_line_and_tab_in_comments():
    for pair in code_comments_list:
        str_text = "%r" % pair.comment.useful_text
        pair.comment.useful_text = str_text[1:-1]


def export_overview_to_txt_file():
    file_overview = codecs.open('data/overview/overview.txt', 'a+', 'utf-8')
    text_to_write = ''
    for pair in code_comments_list:
        text_to_write += 'PHP' + '\t' + pair.repo_desc + '\t' + pair.source_desc + '\t' + str(pair.pair_id) + '\t' + pair.comment.useful_text + '\n'

    file_overview.write(text_to_write)
    file_overview.close()


def create_dict_of_pairs():
    list_of_dict = []
    for pair in code_comments_list:
        curr_dict = {
            'pair_id': pair.pair_id,
            'comment': pair.comment.useful_text,
            'code': pair.code.replace('\n', '').replace('\t', ''),
        }
        for query in read_queries('translated_updated.txt'):
            curr_dict[query] = 0
        list_of_dict.append(curr_dict)

    return list_of_dict


def read_queries(path='translated_updated.txt'):
    """
    Gets list of queries from the file

    :param path: Path to the file containing list of queries (one query per line)
    :return: list of queries
    """

    with open(path, 'r') as file:
        queries = file.readlines()

    return queries


def export_data_to_csv():
    list_of_dict = create_dict_of_pairs()
    df = pd.DataFrame(list_of_dict)
    df.to_csv('data/annotation_matrix.csv', index=False, mode='a')


def export_data():
    prepare_data_for_export()
    export_pairs_to_txt_files()
    export_overview_to_txt_file()
    export_data_to_csv()

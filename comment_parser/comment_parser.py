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


class Code_comment:
    def __init__(self, repo_desc, source_desc, code, comment: Comment):
        self.pair_id = None
        self.repo_desc = repo_desc
        self.source_desc = source_desc
        self.code = code
        self.comment: Comment = comment


code_comments_list: List[Code_comment] = []


def extract_comments():
    php_dict = read_json('php_files.json')

    for key, values in php_dict.items():
        for value in values:
            extract_comments_from_file(key, value)

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


def extract_comments_from_file(repo_desc, source_desc):
    comment_text = ''
    useful_comment_text = ''

    regex_pattern_start_multiline = r'^([\s]*)\/\*'
    regex_pattern_multiline = r'^([\s]*)\*'
    regex_pattern_end_multiline = r'^([\s]*)\*\/'
    regex_pattern_single = r'^([\s]*)(\#|\/\/)'

    url_php_file = 'https://github.com' + source_desc
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
                extract_code(lines, line_no, comment_text, repo_desc, useful_comment_text, source_desc)

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
        code_comments_list[i].pair_id = repo_part[0] + '_' + repo_part[1] + '_' + source_part[0][:-4] + '_' + str(i + 1)


def export_pairs_to_txt_files():
    for pair in code_comments_list:
        file_code_comment_pair = codecs.open('data/pairs/' + pair.pair_id + '.txt', 'w+', 'utf-8')
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
        dict = {'pair_id': pair.pair_id, 'comment': pair.comment.useful_text, 'code': pair.code.replace('\n', '').replace('\t', ''), 'query': 0, 'convert int to string,priority queue': 0, 'string to date': 0, 'sort string list': 0, 'save list to file': 0, 'postgresql connection': 0, 'confusion matrix': 0, 'set working directory': 0, 'group by count': 0, 'binomial distribution': 0, 'aes encryption': 0, 'linear regression': 0, 'socket recv timeout': 0, 'write csv': 0, 'convert decimal to hex': 0, 'export to excel': 0, 'scatter plot': 0, 'convert json to csv': 0, 'pretty print json': 0, 'replace in file': 0, 'k means clustering': 0, 'connect to sql': 0, 'html encode string': 0, 'finding time elapsed using a timer': 0, 'parse binary file to custom class': 0, 'get current ip address': 0, 'convert int to bool': 0, 'read text file line by line': 0, 'get executable path': 0,
                'httpclient post json': 0, 'get inner html': 0, 'convert string to number': 0, 'format date': 0, 'readonly array': 0, 'filter array': 0, 'map to json': 0, 'parse json file': 0, 'get current observable value': 0, 'get name of enumerated value': 0, 'encode url': 0, 'create cookie': 0, 'how to empty array': 0, 'how to get current date': 0, 'how to make the checkbox checked': 0, 'initializing array': 0, 'how to reverse a string': 0, 'read properties file': 0, 'copy to clipboard': 0, 'convert html to pdf': 0, 'json to xml conversion': 0, 'how to randomly pick a number': 0, 'normal distribution': 0, 'nelder mead optimize': 0, 'hash set for counting distinct elements': 0, 'how to get database table name': 0, 'deserialize json': 0, 'find int in string': 0, 'get current process id': 0, 'regex case insensitive': 0, 'custom http error response': 0,
                'how to determine a string is a valid word': 0, 'html entities replace': 0, 'set file attrib hidden': 0, 'sorting multiple arrays based on another arrays sorted order': 0, 'string similarity levenshtein': 0, 'how to get html of website': 0, 'buffered file reader read text': 0, 'encrypt aes ctr mode': 0, 'matrix multiply': 0, 'print model summary': 0, 'unique elements': 0, 'extract data from html content': 0, 'heatmap from 3d coordinates': 0, 'get all parents of xml node': 0, 'how to extract zip file recursively': 0, 'underline text in label widget': 0, 'unzipping large files': 0, 'copying a file to a path': 0, 'get the description of a http status code': 0, 'randomly extract x items from a list': 0, 'convert a date string into yyyymmdd': 0, 'convert a utc time to epoch': 0, 'all permutations of a list': 0, 'extract latitude and longitude from given input': 0,
                'how to check if a checkbox is checked': 0, 'converting uint8 array to image': 0, 'memoize to disk - persistent memoization': 0, 'parse command line argument': 0, 'how to read the contents of a .gz compressed file?': 0, 'sending binary data over a serial connection': 0, 'extracting data from a text file': 0, 'positions of substrings in string': 0, 'reading element from html - <td>': 0, 'deducting the median from each column': 0, 'concatenate several file remove header lines': 0, 'parse query string in url': 0, 'fuzzy match ranking': 0, 'output to html file': 0}
        list_of_dict.append(dict)

    return list_of_dict


def export_data_to_csv():
    list_of_dict = create_dict_of_pairs()
    df = pd.DataFrame(list_of_dict)
    df.to_csv('data/annotation_matrix.csv', index=False, mode='a')


def export_data():
    prepare_data_for_export()
    export_pairs_to_txt_files()
    export_overview_to_txt_file()
    export_data_to_csv()

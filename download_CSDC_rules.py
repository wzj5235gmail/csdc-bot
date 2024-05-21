# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 22:28:24 2024

@author: Administrator
"""

import requests
import re

headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
}
fail_list = []

def get_html(url):
    r = requests.get(url, headers=headers)
    r.encoding = r.apparent_encoding
    html = r.text
    return html

def main():
    domain = 'http://www.chinaclear.cn'
    url = 'http://www.chinaclear.cn/zdjs/fzhgl/law_flist.shtml'
    html = get_html(url)
    page_urls = re.findall('/.*law_flist.shtml', html)
    for page_url in page_urls[2:]:
        section_urls = get_section_urls(domain + page_url)
        for section_url in section_urls:
            section_html = get_section_html(domain + section_url)
            pdf_list = get_pdf_list(section_html)
            for url in pdf_list:
                if url[:4] == 'http':
                    download(url)
                else:
                    download(domain + url)

def get_section_urls(url):
    page_html = get_html(url)
    section_urls = re.findall('/.*law_list.shtml', page_html)
    section_urls = [i[:-6] + '/code_2.shtml' for i in section_urls]
    return section_urls

def get_section_html(url):
    section_html = get_html(url)
    return section_html

def get_pdf_list(html):
    pdf_list = re.findall('f=".*.pdf', html)
    pdf_list = [i[3:] for i in pdf_list]
    return pdf_list

def download(url):
    try:
        response = requests.get(url, headers=headers)
        name = re.findall('file[s]?/.*.pdf', url)[0][6:]
        with open('/Users/danieljames/Downloads/rules/new/' + name, 'wb') as file:
            file.write(response.content)
        print("文件下载成功！")
    except:
        fail_list.append(url)
        print(f"下载失败: {url}")

main()

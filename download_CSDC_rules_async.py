# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 22:28:24 2024

@author: Administrator
"""

import aiohttp
import aiofiles
import asyncio
import re

headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
}
fail_list = []

async def get_html(session, url):
    async with session.get(url, headers=headers) as response:
        html = await response.text()
        return html

async def get_section_urls(session, url):
    page_html = await get_html(session, url)
    section_urls = re.findall('/.*law_list.shtml', page_html)
    section_urls = [i[:-6] + '/code_2.shtml' for i in section_urls]
    return section_urls

async def get_pdf_list(html):
    pdf_list = re.findall('f=".*.pdf', html)
    pdf_list = [i[3:] for i in pdf_list]
    return pdf_list

async def download(session, url):
    try:
        async with session.get(url, headers=headers) as response:
            name = re.findall('file[s]?/.*.pdf', url)[0][6:]
            async with aiofiles.open('/Users/danieljames/Downloads/rules/new/' + name, 'wb') as file:
                content = await response.read()
                await file.write(content)
        print("文件下载成功！")
    except:
        fail_list.append(url)
        print(f"下载失败: {url}")

async def main():
    domain = 'http://www.chinaclear.cn'
    url = 'http://www.chinaclear.cn/zdjs/fzhgl/law_flist.shtml'
    async with aiohttp.ClientSession() as session:
        html = await get_html(session, url)
        page_urls = re.findall('/.*law_flist.shtml', html)
        for page_url in page_urls[2:]:
            section_urls = await get_section_urls(session, domain + page_url)
            for section_url in section_urls:
                section_html = await get_html(session, domain + section_url)
                pdf_list = await get_pdf_list(section_html)
                tasks = []
                for url in pdf_list:
                    if url[:4] == 'http':
                        tasks.append(download(session, url))
                    else:
                        tasks.append(download(session, domain + url))
                await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())
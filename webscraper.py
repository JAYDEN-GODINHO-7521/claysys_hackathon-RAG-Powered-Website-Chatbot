from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from bs4 import BeautifulSoup

def main(website_url):
    
    url = website_url+'/sitemap.xml'
    urlss=[]
    flag=1
    while(flag):
        try:
            reqs = requests.get(url)
            soup = BeautifulSoup(reqs.text, 'xml')
            urls = [url.text for url in soup.find_all('loc') if website_url in url.text]
            urlss+=urls
        except requests.exceptions.RequestException as e:
            print(f"Error fetching sitemap: {e}")
            exit()
        except Exception as e:
            print(f"Error parsing sitemap: {e}")
            exit()
        for urll in urlss:
            if urll.endswith('xml'):
                url=urll
                urlss.pop(0) 
                break
            else:
                flag=0
    nos_of_web_pages=(f"Found {len(urlss)} URLs from the sitemap to start crawling.")


    def get_html(url):

        headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
            ("h4", "Header 4"),
        ]

        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        # for local file use html_splitter.split_text_from_file(<path_to_file>)
        html_header_splits = html_splitter.split_text_from_url(url)

        chunk_size = 500
        chunk_overlap = 30
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Split
        splits = text_splitter.split_documents(html_header_splits)

        return splits


    emb={}
    for url in urlss:
        splits=get_html(url)
        for i in range(len(splits)):
            values = list(splits[i].metadata.values())
            result_string = ", ".join(values)
            emb[i]=result_string+" "+splits[i].page_content


    strings = []
    for i in range(len(emb)):
        string=Document(
                    page_content=emb[i],
                )
        strings.append(string)

    embeddings = OpenAIEmbeddings(check_embedding_ctx_length=False,  openai_api_key="sk-1234", base_url="http://localhost:8080/v1",model="text-embedding-nomic-embed-text-v1.5")

    vectorstore = Chroma.from_documents(documents=strings, 
                                        embedding=embeddings,
                                        persist_directory="chroma_persist")

    return nos_of_web_pages

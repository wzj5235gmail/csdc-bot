import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.vectorstores import FAISS
import dotenv
from langchain_text_splitters import CharacterTextSplitter


dotenv.load_dotenv()
embed = QianfanEmbeddingsEndpoint()


def get_docs(path):
    pages = []
    for root, dirs, files in os.walk(path):
        for name in files:
            try:
                loader = PyPDFLoader(os.path.join(root, name))
                pages.extend(loader.load())
            except:
                print("Fail:", name)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    return docs

if __name__ == "__main__":
    path = '/Users/danieljames/Downloads/rules/embed'
    docs = get_docs(path)
    print(f"{len(docs)} file to process")
    success_count = 0
    fail_count = 0
    vectors = FAISS.from_documents([docs[0]], QianfanEmbeddingsEndpoint(model="bge_large_zh", endpoint="bge_large_zh"))
    for doc in docs[1:]:
        try:
            vectors.add_documents([doc])
            success_count += 1
            print("success count:", success_count)
        except Exception as e:
            fail_count += 1
            print(e)
            with open('errors.log', 'w+') as f:
                f.write(doc.__repr__() + '\n')
    vectors.save_local('vectors')
    print(f'All Done. Success count: {success_count} Fail count: {fail_count}')

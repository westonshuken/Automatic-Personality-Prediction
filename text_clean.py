import re
from nltk.corpus import stopwords

class preprocess:

    def url_remove(self, post):
        return re.sub(r'http\S+', '', post)

    def pipe_remove(self, post):
        return re.sub(r'[|]', ' ', post)

    def punc_remove(self, post):
        return re.sub(r'[\'_:]', '', post)

    def remove_dig_token(self, post):
        return [post[i] for i in range(len(post)) if post[i].isalpha()]

    def remove_stopwords(self, post):
        sw = stopwords.words('english')
        return [post[i] for i in range(len(post)) if post[i] not in sw]
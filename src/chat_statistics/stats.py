import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Union

import arabic_reshaper
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
from hazm import Normalizer, sent_tokenize, word_tokenize
from loguru import logger
from src.data import DATA_DIR
from wordcloud import WordCloud


class ChatStatistics:
    """Generating chat statistics from a telegram chat json file
    """
    def __init__(self, chat_json: Union[str, Path]):
        """Generates a word cloud from the chat data 
         
        Args:
            chat_json : path to telegram export json file
        """
        #load chat data
        logger.info(f"Loading chat data from {chat_json}")
        with open(chat_json) as f:
            self.chat_data=json.load(f)

        #load stopwords    
        self.normalizer = Normalizer()
        logger.info(f"Loading stopwords data from {DATA_DIR / 'stop_words.txt'}")
        self.stop_words= open(DATA_DIR / 'stop_words.txt').readlines()
        self.stop_words= set(map(str.strip,self.stop_words))
        self.stop_words= set(map(self.normalizer.normalize, self.stop_words))

    @staticmethod   
    def rebuild_msg(sub_message):
        msg_text=''
        for sub_msg in sub_message:
            if isinstance(sub_msg, str):
                msg_text += sub_msg
            elif  'text' in sub_msg:
                msg_text +=sub_msg['text'] 
        return msg_text
    
    def msg_has_question(self,msg):
        if not isinstance(msg['text'], str):
            msg['text'] = self.rebuild_msg(msg['text'])
        sentences = sent_tokenize(msg['text'])
        for sentence in sentences:
            if ('?' not in sentence) and ('؟' not in sentence):
                continue
            return True
    
    def get_top_users(self, top_n=10):
        """[summary]
        generate statistics from the chat data
        """
        is_question = defaultdict(bool)
        for msg in self.chat_data['messages']:
            if not isinstance(msg['text'], str):
                msg['text'] = self.rebuild_msg(msg['text'])
            sentences = sent_tokenize(msg['text'])
            for sentence in sentences:
                if ('?' not in sentence) and ('؟' not in sentence):
                    continue
                is_question[msg['id']]=True

        logger.info("Getting top users...")
        users = []

        for message in self.chat_data['messages']:
            if not message.get('reply_to_message_id'):
                continue
            if is_question[message['reply_to_message_id']] is False:
                continue
            users.append(message['from'])
        return Counter(users).most_common(top_n)

    def generate_word_cloud(
        self,
        output_dir :Union [str, Path],
        width: int = 800, height: int = 600,
        max_font_size: int = 250,
        background_color: str = 'white',
        ):
        """Generates a word cloud from the chat data 

        Args:
            output_dir : path to output directory for word cloud image
        """
        logger.info(f"Generating word cloud...")
        text_content=''
        for msg in self.chat_data['messages']:
            if type(msg['text']) is str:
                tokens= word_tokenize(msg['text'])
                tokens = list(filter(lambda item:  item not in self.stop_words,tokens))
                text_content += f"{' '.join(tokens)}"
        
        text_content = get_display(text_content)
        #text_content = arabic_reshaper.reshape(text_content)

        logger.info(f"saving word cloud to {output_dir}")
        wordcloud=WordCloud(
        width=width,
        height=height,
        font_path=str(DATA_DIR / 'BHoma.ttf'),
        max_font_size=200,
        background_color='white'
        ).generate(text_content)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        wordcloud.to_file(str(Path(output_dir) / 'wordcloud.png'))
if __name__== "__main__":
    chat_stats = ChatStatistics(chat_json=DATA_DIR / 'online.json')
    chat_stats.generate_word_cloud(output_dir=DATA_DIR)
    top_users = chat_stats.get_top_users(top_n=10)
    print(top_users)
print('Done!')

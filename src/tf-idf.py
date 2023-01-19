from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer


def extract_keywords_tf_idf(corpus: List[str], num_keywords: int = 10) -> List[str]:
    """
    Extract keywords using TF-IDF
    :param corpus: Corpus of text
    :param num_keywords: Number of keywords to extract
    :return: List of keywords
    """
    tfidf = TfidfVectorizer().fit_transform(corpus)

    # Get the average TF-IDF score for each word
    average_tfidf = np.mean(tfidf, axis=0).tolist()[0]

    # Sort the words by their TF-IDF score
    sorted_indices = np.argsort(average_tfidf)[::-1]

    # Extract the keywords
    extracted_keywords = [corpus[0].split()[i] for i in sorted_indices[:num_keywords]]

    return extracted_keywords


if __name__ == "__main__":
    prompt = [
        """\"나는 근본적으로 먹고사는 것에 대한 두려움을 가지고 있다. 어쩌면 이것은 가난에 대한 트라우마라고 할 수도 있다. 가난을 겪어보니 다시 그것을 겪는다고 생각하면 몸서리가 나는 것이다. 
        시대가 고만고만할 때는 모두가 가난했다. 너도 나도 흠이 없었다. 방이 한 개여도, 화장실이 밖에 있고 구식이어도. 모든 집이 그러했을 때였다. 그래서일까. 그때는 가난하다는 생각을 해본 적이 
        없다. 가난은 세월이 흐르면서 점점 더 커졌다. 아이러니하게도, 예전보다 집이 커지고 벌이는 늘어나는데 가난에 대한 두려움은 더 커진 것이다. 화장실도 두 개나 있고 신식인데도. 나는 왜 이리 
        넉넉하지 못하다는 마음으로 오늘 하루를 또 무겁게 지내고 있는 것일까? 가난은 상대적이기 때문이다. 이전보다 나아졌다곤 하지만, 나보다 더 큰 집을 가진 사람을 바라보고 더 높은 급여를 받는 
        사람들을 우러러본다. 더 좋은 차와 노후에도 걱정 없을 건물을 가진 사람들에 비해 나는 한없이 작아지고 가난해지는 것이다. 더불어, 가난은 트라우마이기 때문이다. 물에 빠져 죽을 뻔한 사람이 물을 
        무서워하듯이. 가난에 익사할 뻔한 기억이 있는 사람에게 또다시 가난해질지 모른다는 두려움은 사람을 무기력하게 만든다. 월급이 끊겨 한 달씩 내야 하는 휴대폰 비용이나 공과금을 내지 못하면 
        어떡하나. 나 하나 잘못되어 우리 가족이 힘들어지면 어쩌나... 꼬리에 꼬리를 무는 걱정을 하다 보면 그 끝에서 나는 가난의 트라우마에 압도당한다. 그러나 나는 정말 내가 가진 문제를 발견하게 
        되었다. 그건 바로 '가난한 마음'이다. 마음이 가난하니 생활이 가난해지는 것이다. 삶 자체가 부유하지 못한 것이다. 이걸 깨달은 지 얼마 안 되었다는 걸 생각해보면, 가난이란 얼룩이 얼마나 
        짙은가에 새삼 놀라고 만다. 마음이 가난하면 사람은 스스로를 옥죈다. 행복의 광활한 들판보다 불행의 틈새에 코를 박고 이러지도 저러지도 못한다. 한껏 고개를 들어 하늘을 보며 숨 한 번 크게 
        들이켜면 되는데, 스스로의 숨을 막고 있는 그 느낌. 마음이 가난하면 아껴야 할 때 아끼지 못하고, 아끼지 말아야 할 때 아끼는 함정에서 헤어 나오질 못한다. 마음이 가난함을 깨닫고, 
        그 마음을 좀 내려놓으니. 이제는 아껴야 할 때를 알고, 아끼지 말아야 할 때를 조금은 더 구분하게 되었다. 이로써, 먹고사는 것에 대한 고단함만을 느끼던 내가. 이제는 그것에 대한 고귀함을 
        논하게 되었다. 먹고사는 것은 가난에서 벗어나려는 발버둥이자, 결국 그로 인해 많은 것을 배우고 내가 채워져 가는 과정이기 때문이다. 가난과 먹고사는 것에 대한 걱정으로 벌벌 떨던 한 존재가, 
        저도 모르게 그 과정을 거치며 성장해 있는 모습을 알아차리는 것. 자산의 크기와 액수를 떠나, 채워진 그 마음이라면 더 이상 가난하지 않을 수 있다는 용기와 위로의 마음. 가난한 자의 마음은 
        돈으로도 어찌할 수가 없다. 먹고사는 고단함에서 고귀함을 찾아낸다면, 마음은 가난에서 벗어날 수 있을 것이다. 가난이라는 무기력함에 빠져들 때, 통장이 아니라 마음을 먼저 열어보는 이유다. 
        스테르담 저서, 강의, 프로젝트 스테르담 인스타그램"""]

    tokenizer = AutoTokenizer.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")

    tokenized = tokenizer(prompt).input_ids[0]

    prompt = [" ".join(map(str, tokenized))]

    # Print result
    keywords = extract_keywords_tf_idf(prompt, num_keywords=10)
    print(f"Keywords: {keywords}")

    keywords = list(map(int, keywords))

    print(" ".join(map(tokenizer.decode, keywords)))

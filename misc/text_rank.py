import networkx as nx
import numpy as np
 
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
 
def textrank(document):
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(document)
 
    bow_matrix = CountVectorizer().fit_transform(sentences)
    normalized = TfidfTransformer().fit_transform(bow_matrix)
 
    similarity_graph = normalized * normalized.T
 
    nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
    scores = nx.pagerank(nx_graph)
    return sorted(((scores[i],s) for i,s in enumerate(sentences)),
                  reverse=True)


# document = """To Sherlock Holmes she is always the woman. I have
# seldom heard him mention her under any other name. In his eyes she
# eclipses and predominates the whole of her sex. It was not that he
# felt any emotion akin to love for Irene Adler. All emotions, and that
# one particularly, were abhorrent to his cold, precise but admirably
# balanced mind. He was, I take it, the most perfect reasoning and
# observing machine that the world has seen, but as a lover he would
# have placed himself in a false position. He never spoke of the softer
# passions, save with a gibe and a sneer. They were admirable things for
# the observer-excellent for drawing the veil from men’s motives and
# actions. But for the trained reasoner to admit such intrusions into
# his own delicate and finely adjusted temperament was to introduce a
# distracting factor which might throw a doubt upon all his mental
# results. Grit in a sensitive instrument, or a crack in one of his own
# high-power lenses, would not be more disturbing than a strong emotion
# in a nature such as his. And yet there was but one woman to him, and
# that woman was the late Irene Adler, of dubious and questionable
# memory.
# """
document = "The contest was really unbalanced consisting of two easy problems and three difficult problems due to which some deserving teams ended up getting not so good ranks. If you are from a college with fairly decent coding culture, a few wrong submissions could lead to your team not getting selected for the regionals. Having said that, I think most of the good teams would have solved at least the first two questions in an hour and they will most probably qualify for the regionals. I also feel that very few people solving any of the last three questions in the first two hours may be a reason why these problems got such a few accepted solutions. Most of the teams including us thought they might be too tricky for us to solve. But I guess a good number of teams could have solved the third question given the fact that it was pretty much brute force."


result = textrank(document)
for (score,sent) in result:
	print(sent)
	print()

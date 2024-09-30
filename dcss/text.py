import itertools
import re
from collections import Counter
from glob import glob
from itertools import chain, combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.matutils import corpus2csc
from gensim.models.phrases import Phrases, Phraser
import spacy
from spacy.symbols import NOUN, PROPN, VERB
from spacy.training import Example
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')


def bigram_process(
    texts, nlp=None, threshold=0.75, scoring="npmi", detokenize=True, n_process=1
):
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

    sentences = []
    docs = []

    # sentence segmentation doesn't need POS tagging or lemmas.
    for doc in tqdm(
        nlp.pipe(texts, disable=["tagger", "lemmatizer", "ner"], n_process=n_process),
        total=len(texts),
        desc="Processing sentences",
    ):
        doc_sents = [
            [
                token.text.lower()
                for token in sent
                if token.text != "\n" and token.is_alpha
            ]
            for sent in doc.sents
        ]
        sentences.extend(doc_sents)
        docs.append(doc_sents)

    model = Phrases(sentences, min_count=1, threshold=threshold, scoring=scoring)
    bigrammer = Phraser(model)
    bigrammed_list = [
        [bigrammer[sent] for sent in doc] 
        for doc in tqdm(docs, desc="Detecting bigrams")
    ]

    if detokenize:
        bigrammed_list = [[" ".join(sent) for sent in doc] for doc in bigrammed_list]
        bigrammed_list = [" ".join(doc) for doc in bigrammed_list]
    elif detokenize == "sentences":
        bigrammed_list = [[" ".join(sent) for sent in doc] for doc in bigrammed_list]
    else:
        bigrammed_list = list(chain(*bigrammed_list))

    return model, bigrammed_list


def preprocess(
    texts, nlp=None, bigrams=False, detokenize=True, n_process=1, custom_stops=[]
):
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

    processed_list = []
    allowed_postags = [92, 96, 84]  # 'NOUN', 'PROPN', 'ADJ'

    if bigrams:
        model, texts = bigram_process(texts, detokenize=True, n_process=n_process)

    for doc in tqdm(
        nlp.pipe(texts, disable=["ner", "parser"], n_process=n_process),
        total=len(texts),
        desc="Preprocessing documents",
    ):
        processed = [
            token.lemma_
            for token in doc
            if not token.is_stop and len(token) > 1 and token.pos in allowed_postags
        ]

        if detokenize:
            processed = " ".join(processed)
            processed_list.append(processed)
        else:
            processed_list.append(processed)

    if bigrams:
        return model, processed_list
    else:
        return processed_list



def see_semantic_context(search, text, window):
    keysearch = re.compile(search, re.IGNORECASE)
    contexts = []

    tokens = text.split()
    tokens = [t.lower() for t in tokens]
    token_count = Counter(tokens)

    for index in range(len(tokens)):
        if keysearch.match(tokens[index]):
            start = max(0, index-window)
            finish = min(len(tokens), index+window+1)
            left = " ".join(tokens[start:index])
            right = " ".join(tokens[index+1:finish])
            contexts.append("{} **{}** {}".format(left, tokens[index].upper(), right))

    return contexts, token_count[search]


def bow_to_df(gensim_corpus, gensim_vocab, tokens_as_columns=True):
    csr = corpus2csc(gensim_corpus)
    df = pd.DataFrame.sparse.from_spmatrix(csr)
    if tokens_as_columns is True:
        df = df.T
    df.columns = [v for k,v in gensim_vocab.items()]
    return df


def get_topic_words(vectorizer, model, n_top_words=10):
    """
    Given a vectorizer object and a fit model (e.g. LSA, NMF, LDA) from sklearn
    Gets the top words associated with each and returns them as a dict.
    """
    words = vectorizer.get_feature_names_out()

    topics = {}
    for i,topic in enumerate(model.components_):
        topics[i] = " ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return topics

def get_topic_word_scores(df, n_words, topic_column, all_topics=False, as_tuples=True):
    df = df.sort_values(by=[topic_column], ascending = False)
    if all_topics is True:
        result = pd.concat([df.head(n_words), df.tail(n_words)]).round(4)
    else:
        result = pd.concat([df.head(n_words), df.tail(n_words)]).round(4)
        result = result[topic_column]
        if as_tuples is True:
            result = list(zip(result.index, result))
    return result 


def topic_to_dataframe(model, topic):
    topic = model.show_topic(topic, topn=30)
    df = pd.DataFrame(topic, columns = ['Word', 'Probability'])
    return df

def plot_topic(model, topic):
    df = topic_to_dataframe(model, topic)

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Probability', y='Word',
                    color='lightgray')

    for i in zip(df['Probability'], df['Word']):
        plt.text(x=i[0], y=i[1], s=i[1], fontsize=6)

    ax.set(yticklabels=[], ylabel='',
           xlabel='Probability',
           title=f'Topic {topic}')
    sns.despine(left=True)
    plt.show()







#CHAPTER 33 FUNCTIONS


def create_examples(text):

    examples = []
    for sent in nlp(text).sents:
        labels = {'entities':[]}
        sent_doc = sent.as_doc()
        for token in sent_doc:

            if token.text == "Cambridge" and token.nbor(1).text == "Analytica":
                label = (token.idx, token.nbor(1).idx + len(token.nbor(1)), "ORG")
                labels['entities'].append(label)

            if token.text == "Facebook":
                label = (token.idx, token.idx + len(token), "ORG")
                labels['entities'].append(label)

            if token.text == "Wylie":
                label = (token.idx, token.idx + len(token), "PERSON")
                labels['entities'].append(label)

        if len(labels['entities']) > 0:
            example_sent = nlp.make_doc(sent.text)
            examples.append(Example.from_dict(example_sent, labels))

    return examples


def entity_sentiment(doc, classifier, entity_names = [], entity_types = []):
    sentence_list = []
    entities_list = []
    sentiment_list = []
    sentiment_score_list = []
    sent_start_list = []
    sent_end_list = []

    for sent in doc.sents:
        entities = []
        for ent in sent.ents:
            if len(entity_names) > 0 and len(entity_types) > 0:
                for entity in entity_names:
                    if ent.text == entity and ent.label_ in entity_types:
                        entities.append(ent.text)
            elif len(entity_types) > 0:
                if ent.label_ in entity_types:
                    entities.append(ent.text)
            else:
                entities.append(ent.text)
        if len(entities) > 0:
                sentence_list.append(sent.text)
                sent_start_list.append(sent.start)
                sent_end_list.append(sent.end)
                entities_list.append(entities)
                sentiment = classifier(sent.text)
                sentiment_list.append(sentiment[0]['label'])
                sentiment_score_list.append(sentiment[0]['score'])

    df = pd.DataFrame()
    df['sentence'] = sentence_list
    df['entities'] = entities_list
    df['sentiment'] = sentiment_list
    df['sentiment_score'] = sentiment_score_list
    df['sent_signed'] = df['sentiment_score']
    df.loc[df['sentiment'] == 'NEGATIVE', 'sent_signed'] *= -1
    df['sentence_start'] = sent_start_list
    df['sentence_end'] = sent_end_list

    return df

def process_speeches_sentiment(df, nlp, sentiment):
    ent_types = ['GPE', 'ORG', 'PERSON']
    speakers = df['speakername'].tolist()
    speeches = df['speechtext'].tolist()
    speeches = [str(s).replace('\n',' ').replace('\r', '') for s in speeches]
    speeches_processed = [speech for speech in nlp.pipe(speeches)]
    print("Speeches processed, analyzing sentiment.")
    speaker_dfs = []
    for speaker, speech in zip(speakers, speeches_processed):
        temp_df = entity_sentiment(speech, sentiment, entity_types = ent_types)
        temp_df['speaker'] = speaker
        speaker_dfs.append(temp_df)

    new_df = pd.concat(speaker_dfs)

    return new_df


def create_speaker_edge_df(df, speaker):
    df['entities'] = df['entities'].map(set)
    df = df[df['entities'].map(len) > 1].copy()
    df = df[df['speaker'] == speaker].copy().reset_index(drop=True)

    temp = [[i,] + sorted(y) for i, x in df['entities'].items() for y in combinations(x,2)]
    temp_df = pd.DataFrame(temp, columns=['idx','source','target']).set_index('idx')
    df = temp_df.merge(df['sent_signed'], how='left', left_index=True, right_index=True).reset_index(drop=True)
    df.rename(columns={'sent_signed':'weight'}, inplace = True)

    df = df.sort_values(by='weight', ascending = False)

    return df

def shrink_sent_df(df):
    df = df.groupby(['source','target'])['weight'].agg(
        weight='count').reset_index()

    df = df.sort_values(by='weight', ascending = False)

    return df




def get_sentiment_blocks_df(G, state):

    try:
        import graph_tool as gt
    except:
        print("Error importing graph-tool. Make sure that it's correctly installed.")

    df = pd.DataFrame()
    ent_list = []
    block_list = []

    levels = state.get_levels()
    base_level = levels[0].get_blocks()

    for v in G.vertices():
        ent_list.append(G.vp['labels'][v])
        block_list.append(base_level[v])
    df['entity'] = ent_list
    df['block'] = block_list

    return df

def calculate_avg_block_sentiment(results_df, edges_df):

    block_df = results_df.groupby(['block'])['entity'].agg(entities=list).reset_index()

    block_df = block_df[block_df['entities'].map(len) > 1].copy()

    avg_sentiments = []

    ent_lists = block_df['entities'].tolist()

    for ent_list in ent_lists:
        sentiments = []
        sorted_combinations = [sorted(y) for y in combinations(ent_list,2)]
        for pair in sorted_combinations:
            sentiment = edges_df[(edges_df['source'] == pair[0]) & (edges_df['target'] == pair[1])].weight.mean()
            sentiments.append(sentiment)
        avg_sentiments.append(np.nanmean(sentiments))
    block_df['avg_sentiments'] = avg_sentiments

    return block_df



# SVO FUNCTIONS

SUBJ_DEPS = {'agent', 'csubj', 'csubjpass', 'expl', 'nsubj', 'nsubjpass'}
OBJ_DEPS = {'attr', 'dobj', 'dative', 'oprd'}
AUX_DEPS = {'aux', 'auxpass', 'neg'}

def _get_conjuncts(tok):
    """
    Function from Textacy.
    
    Return conjunct dependents of the leftmost conjunct in a coordinated phrase,
    e.g. "Burton, [Dan], and [Josh] ...".
    """
    return [right for right in tok.rights
            if right.dep_ == 'conj']

def get_main_verbs_of_sent(sent):
    """Return the main (non-auxiliary) verbs in a sentence."""
    return [
        tok for tok in sent
        if tok.pos == VERB and tok.dep_ not in {'aux', 'auxpass'}
    ]


def get_subjects_of_verb(verb):
    """
    Function from Textacy.
    
    Return all subjects of a verb according to the dependency parse.
    """
    subjs = [tok for tok in verb.lefts if tok.dep_ in SUBJ_DEPS]
    # get additional conjunct subjects
    subjs.extend(tok for subj in subjs for tok in _get_conjuncts(subj))
    return subjs


def get_objects_of_verb(verb):
    """
    Function from Textacy.
    
    Return all objects of a verb according to the dependency parse,
    including open clausal complements.
    """
    objs = [tok for tok in verb.rights if tok.dep_ in OBJ_DEPS]
    # get open clausal complements (xcomp)
    objs.extend(tok for tok in verb.rights if tok.dep_ == 'xcomp')
    # get additional conjunct objects
    objs.extend(tok for obj in objs for tok in _get_conjuncts(obj))
    return objs


def get_span_for_verb_auxiliaries(verb):
    """
    Function from Textacy.
    
    Return document indexes spanning all (adjacent) tokens
    around a verb that are auxiliary verbs or negations.
    """
    min_i = verb.i - sum(1 for _ in itertools.takewhile(
        lambda x: x.dep_ in AUX_DEPS, reversed(list(verb.lefts))))
    max_i = verb.i + sum(1 for _ in itertools.takewhile(
        lambda x: x.dep_ in AUX_DEPS, verb.rights))
    return (min_i, max_i)


def get_span_for_compound_noun(noun):
    """
    Function from Textacy.
    
    Return document indexes spanning all (adjacent) tokens
    in a compound noun.
    """
    min_i = noun.i - sum(1 for _ in itertools.takewhile(
        lambda x: x.dep_ == 'compound', reversed(list(noun.lefts))))
    return (min_i, noun.i)


def subject_verb_object_triples(doc):
    """
    Modified version of the function from Textacy.
    """
    sents = doc.sents

    for sent in sents:
        start_i = sent[0].i

        verbs = get_main_verbs_of_sent(sent)
        for verb in verbs:
            subjs = get_subjects_of_verb(verb)
            if not subjs:
                continue
            objs = get_objects_of_verb(verb)
            if not objs:
                continue

            # add adjacent auxiliaries to verbs, for context
            # and add compounds to compound nouns
            verb_span = get_span_for_verb_auxiliaries(verb)
            verb = sent[verb_span[0] - start_i:verb_span[1] - start_i + 1]
            for subj in subjs:
                subj = sent[get_span_for_compound_noun(subj)[0] -
                            start_i:subj.i - start_i + 1]
                for obj in objs:
                    if obj.pos == NOUN:
                        span = get_span_for_compound_noun(obj)
                    elif obj.pos == VERB:
                        span = get_span_for_verb_auxiliaries(obj)
                    else:
                        span = (obj.i, obj.i)
                    obj = sent[span[0] - start_i:span[1] - start_i + 1]

                    yield (subj, verb, obj)

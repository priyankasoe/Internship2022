from webbrowser import get
import wikipediaapi 
import pandas as pd
import concurrent.futures
from tqdm import tqdm
import re
import spacy
import neuralcoref
import requests
import time 
from pathlib import Path
from bs4 import BeautifulSoup

class FinalCode:

    def __init__(self, page_name):
        self.page_name = page_name
        self.nlp = spacy.load('en_core_web_sm')
        neuralcoref.add_to_pipe(self.nlp)

    def extract_abstract(self): 
        ''' Extract only abstract from a wikipedia page
        Args: 
            page_name (str): name of the wikipedia page
            
        Returns: 
            String: abstract of wikipedia page'''

        subject = self.page_name
        url = 'https://en.wikipedia.org/w/api.php'
        params = {
                'action': 'query',
                'format': 'json',
                'titles': subject,
                'prop': 'extracts',
                'exintro': True,
                'explaintext': True,
            }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        page = next(iter(data['query']['pages'].values()))

        return page['extract']

    def extract_all_links(self): 
        ''' Extract all the links from a wikipedia page
        Args: 
            page_name (str): name of the wikipedia page
            
        Returns: 
            list (str): containing page titles of all 
            links on original page''' 
        subject = self.page_name
        
        url = 'https://en.wikipedia.org/w/api.php'
        
        params = {
            'action': 'query',
            'format': 'json',
            'titles': subject,
            'prop': 'links',
            'pllimit': 'max',
            'redirects':''
        }
        
        response = requests.get(url=url, params=params)
        data = response.json()
        
        pages = data['query']['pages']
        page = 1
        page_titles = []
        
        try: 
            for key, val in pages.items():
                for link in val['links']:
                    page_titles.append(link['title'])
        except: 
            return page_titles
        
        while 'continue' in data:
            plcontinue = data['continue']['plcontinue']
            params['plcontinue'] = plcontinue
        
            response = requests.get(url=url, params=params)
            data = response.json()
            pages = data['query']['pages']
        
            page += 1
        
            for key, val in pages.items():
                for link in val['links']:
                    page_titles.append(link['title'])
        
        return page_titles

    def extract_text_links(self, abstract_only=False):
        ''' Extract all links from only the text of a wikipedia page
        Args: 
            page_name (str): name of the wikipedia page
            abstract_only (bool): true if only extracting links from just abstract of page, false if all text
            
        Returns: 
            list (str): containing page titles of all links found in text or abstract only of page'''
        subject = self.page_name
        
        url = 'https://en.wikipedia.org/w/api.php'
        params = {
                    'action':'parse',
                    'prop':'text',
                    'format':'json',
                    'page':subject,
                    'section':0,
                    'redirects':''
                }

        if not abstract_only:
            del params['section']

        data = requests.get(url, params=params).json()
        
        soup = BeautifulSoup(data['parse']['text']['*'],'html.parser')

        test = soup.find_all('p')

        a_tags = []
        for paragraph in test: 
            p_tags = paragraph.find_all('a',href=True)
            for tag in p_tags:
                a_tags.append(tag) 
        
        links = []
        for tag in a_tags:
            if not tag.text == '' and 'wiki' in tag['href']:
                links.append(tag['title'])
        
        print(len(links))
        return links

    def extract_contents(self):
        ''' Extract all the contents from a wikipedia page
        Args: 
            page_name (str): name of the wikipedia page
            
        Returns: 
            dataframe: contains page name, text, url, and 
            categories of page'''

        try:
            # get page with wikipedia-api package
            wiki_api = wikipediaapi.Wikipedia(language='en',
                    extract_format=wikipediaapi.ExtractFormat.WIKI)
            page_name = wiki_api.page(self.page_name)

            # check existence to catch errors 
            if not page_name.exists():
                #print('Page {} does not exist.'.format(page_name))
                return

            # creation of pandas dataframe 
            page_data = pd.DataFrame({
                'page': page_name,
                'text': page_name.text,
                'link': page_name.fullurl,
                'categories': [[y[9:] for y in
                            list(page_name.categories.keys())]],
                })

            return page_data
        except:
            print("Page content not found.")
            return None 

    # gets entity pairs in given text 
    def get_entity_pairs(self,text, coref=True):
        ''' Extracts all subject, object, relation triples from given text 
        Args: 
            text (str): all text in wikipedia page 
            
        Returns: 
            dataframe: columns containing subject, object, relation triples, the type of subject, and
            type of object found in the entity'''

        if text is None: 
            return 
        # preprocess text
        text = re.sub(r'\n+', '.', text)  # replace multiple newlines with period
        text = re.sub(r'\[\d+\]', ' ', text)  # remove reference numbers
        text = self.nlp(text)
        if coref:
            text = self.nlp(text._.coref_resolved)  # resolve coreference clusters

        def refine_ent(ent, sent):
            unwanted_tokens = (
                'PRON',  # pronouns
                'PART',  # particle
                'DET',  # determiner
                'SCONJ',  # subordinating conjunction
                'PUNCT',  # punctuation
                'SYM',  # symbol
                'X',  # other
            )
            ent_type = ent.ent_type_  # get entity type
            if ent_type == '':
                ent_type = 'NOUN_CHUNK'
                ent = ' '.join(str(t.text) for t in
                            self.nlp(str(ent)) if t.pos_
                            not in unwanted_tokens and t.is_stop == False)
            elif ent_type in ('NOMINAL', 'CARDINAL', 'ORDINAL') and str(ent).find(' ') == -1:
                refined = ''
                for i in range(len(sent) - ent.i):
                    if ent.nbor(i).pos_ not in ('VERB', 'PUNCT'):
                        refined += ' ' + str(ent.nbor(i))
                    else:
                        ent = refined.strip()
                        break

            return ent, ent_type

        sentences = [sent.string.strip() for sent in text.sents]  # split text into sentences
        ent_pairs = []
        for sent in sentences:
            sent = self.nlp(sent)
            spans = list(sent.ents) + list(sent.noun_chunks)  # collect nodes
            spans = spacy.util.filter_spans(spans)
            with sent.retokenize() as retokenizer:
                [retokenizer.merge(span, attrs={'tag': span.root.tag,
                                                'dep': span.root.dep}) for span in spans]
            deps = [token.dep_ for token in sent]

            for token in sent:
                if str(token) in str(spans):  # only look through sentence objects 
                    subject = [w for w in token.head.lefts if w.dep_
                            in ('subj', 'nsubj')]  # identify subject nodes
                    if subject:
                        subject = subject[0]
                        if str(subject) == str(token):   # when passing through same object 
                            continue
                        # identify relationship by root dependency
                        relation = [w for w in token.ancestors if w.dep_ == 'ROOT']
                        if relation:
                            relation = relation[0]
                            if not relation.is_punct: 
                            # add adposition or particle to relationship
                                try: 
                                    if relation.nbor(1).pos_ in ('ADP', 'PART'):
                                        relation = ' '.join((str(relation), str(relation.nbor(1))))
                                except: 
                                    relation = 'unknown'
                        else:
                            relation = 'unknown'

                        subject, subject_type = refine_ent(subject, sent)
                        token, object_type = refine_ent(token, sent)

                        ent_pairs.append([str(subject), str(relation), str(token),
                                        str(subject_type), str(object_type)])

        ent_pairs = [sublist for sublist in ent_pairs
                            if not any(str(ent) == '' for ent in sublist)]
        pairs = pd.DataFrame(ent_pairs, columns=['subject', 'relation', 'object',
                                                'subject_type', 'object_type'])
        print('Entity pairs extracted:', str(len(ent_pairs)))

        return pairs

    def recursive_get_wiki_pairs(self, k, path, verbose=True):
        ''' Recrusively extracts triples from given page and triples k neighbor sources from the original page
        Args: 
            page_name (str): name of the wikipedia page
            k (int): amount of neighbors to be traversed (ex. if 1 neighbor, entity pairs from the first neighbor sources of 
            the home page would be appended to resulting dataframe)
            path (str): file path to save resulting CSV file to
            verbose (bool): True to see progress of links scraped, False to not see progress 
            
        Returns: 
            dataframe: containing all triples from given page and triples k neighbor sources from the original page
            new CSV file to given directory'''
        start_time = time.time() 

        def wiki_page(link, link_list):
            try:
                if 'Categories' in link or 'Category' in link:   # remove excess sources in all wiki articles 
                    return None, link_list 

                print('Currently scraping: {}'.format(link))
                ents = self.get_entity_pairs(self.extract_abstract(link))

                for source in self.extract_text_links(link, True):
                    link_list.append(source)

                return ents,link_list
            except:
                #print('page is bad')
                return None, link_list

        # create empty ents and sources df 
        #sources = wiki_scrape(page_name)
        sources = self.extract_text_links(self.page_name,True)
        orig_text = self.extract_contents(self.page_name)
        ents = self.get_entity_pairs(orig_text['text'][0])
        len_prev_sources = 0 
        print('Extracted sources and triples from {}!'.format(self.page_name)+' # triples: {}'.format(len(ents)))
        print('Starting recursion')
        count = len(ents) 
        
        for neighbor in range(k): 

            # the new end to sources being looped through is the length of the sources from the past neighbor loop 
            len_curr_neighbor = len(sources) 

            pages = list(sources[len_prev_sources: len_curr_neighbor])
            progress = tqdm(desc='Links Scraped', unit='', total=len(pages)) if verbose else None
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                future_ents = {executor.submit(wiki_page, page,sources): page for page in pages}
                for future in concurrent.futures.as_completed(future_ents):
                    try:
                        data,sources = future.result()
                        if data is not None: 
                            ents = pd.concat([ents,data])
                            count += len(data) 
                    except Exception as e:
                        print('error: {}'.format(e))
                    progress.update(1) if verbose else None     
            progress.close() if verbose else None

            # get the length of the current end of all sources (this includes primary sources, secondary sources, etc)
            # get length of the end of the past neighbor sources (i. e. if currently on neighbor 1 it would get the end of all the first neighbor sources)
            len_prev_sources = len_curr_neighbor

            print('-')
            print('Total ents: {}'.format(len(ents))+' should equal sum of individual ents: {}'.format(count))
            print('done {} neighbor!'.format(neighbor+1))
            print('start next loop: {}'.format(len_prev_sources)) 
            print('end next loop: {}'.format(len(sources))) 
        
        end_time = time.time() 
        print('-')
        print('total scraping time: {} minutes'.format((end_time - start_time) / 60))

        # save dataframe as a CSV file
        try:
            filepath = Path(path+self.page_name+str(k)+'Neighbors'+'EntityPairs.csv')  
            filepath.parent.mkdir(parents=True, exist_ok=True)  
            ents.to_csv(filepath)
        except Exception as e:
            print('Could not turn file into csv. Error: {}'.format(e))

        return ents 



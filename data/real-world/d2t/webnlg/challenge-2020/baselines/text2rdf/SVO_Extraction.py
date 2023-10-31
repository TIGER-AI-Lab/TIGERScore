import regex as re
import numpy as np
import ast
import os
from stanfordcorenlp import StanfordCoreNLP
from lxml import etree
import sys
import requests
from subprocess import getoutput

#Function to convert string to camelCase (for the predicates)
def camelCase(string):
  string = re.sub(r"(_|-)+", " ", string).title().replace(" ", "")
  return string[0].lower() + string[1:]

def Extract_SVO(inputfile, outputfile):
    currentpath = os.getcwd()
    nlp = StanfordCoreNLP(currentpath + '/stanford-corenlp-4.2.2', memory='8g')
    # We will use OpenIE (Open Information Extraction)
    props={'annotators': 'tokenize, ssplit, pos, lemma, depparse, natlog, openie',
           'pipelineLanguage':'en',
           'outputFormat':'json',                        # one of {json, xml, text}
           'openie.format': 'default',    # One of {reverb, ollie, default, qa_srl}
           'openie.triple.strict': 'true',
           'openie.affinity_probability_cap': '1',
           'openie.max_entailments_per_clause': '1000',   # default = 1000
           }

    #Use the translated sentences from the Russian test set
    with open(currentpath + '/' + inputfile, 'rb') as f:
        test = f.readlines()


    newtest = [x.decode('utf-8') for x in test]
    newtext = [re.sub(r'\n', '', x) for x in newtest]

    #Add the first elements
    benchmarkelement = etree.Element("benchmark")
    entrieselement = etree.SubElement(benchmarkelement, "entries")

    for idx, text in enumerate(newtext):
        if not re.search(r'\w', text):
            continue
        entryelement = etree.SubElement(entrieselement, "entry")
        eid = 'Id' + str(idx + 1)
        entryelement.set('eid', eid)
        textelement = etree.SubElement(entryelement, "text")
        textelement.text = text
        gentripelement = etree.SubElement(entryelement, "generatedtripleset")
        try:
            openie = nlp.annotate(text, properties=props)
            openie = ast.literal_eval(openie)              # convert str to dict
        except SyntaxError:
            continue
        print('------------------------------------------------------------------------')
        print('Target Sentence: ' + text)

        #Go over all the triples
        if len(openie["sentences"][0]["openie"]) > 9:
            iteratelength = 8
        else:
            iteratelength = len(openie["sentences"][0]["openie"])
        for j in range(iteratelength):
            svo = openie["sentences"][0]["openie"][j]
            subject = svo['subject']
            subject = re.sub(r' ', '_', subject)
            verb = svo['relation']
            verb = camelCase(verb)
            object = svo['object']
            object = re.sub(r' ', '_', object)
            svostring = subject + ' | ' + verb + ' | ' + object
            print(svostring)
            gtripelement = etree.SubElement(gentripelement, "gtriple")
            gtripelement.text = svostring
            #print((svo['subject'], svo['relation'], svo['object']))

    nlp.close()

    #And save the XML to a file.
    benchmarkelement = etree.tostring(benchmarkelement, encoding="utf-8", xml_declaration=False, pretty_print=True)
    with open(currentpath + '/' + outputfile + '.xml', 'wb') as f:
        f.write(benchmarkelement)

if __name__ == '__main__':
    Extract_SVO(sys.argv[1], sys.argv[2])
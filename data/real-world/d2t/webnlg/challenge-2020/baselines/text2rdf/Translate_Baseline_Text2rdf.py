# Make imports
import time
import pyperclip
from selenium import webdriver
import regex as re
import pickle
import os
import sys

#Function to divide a list into chunks of n-size
def chunks(lst, n):
    chunklist = [lst[i:i + n] for i in range(0, len(lst), n)]
    return chunklist

def TranslateBaseline(inputname, outputname):
    currentpath = os.getcwd()

    #Open the Russian baseline output
    with open(currentpath + '/' + inputname, 'rb') as f:
        baselines = f.readlines()

    #Decode it and remove the \n's.
    baselines = [x.decode('utf-8') for x in baselines]
    baselines = [re.sub(r'\n', '', x) for x in baselines]

    chunkedbaselines = chunks(baselines, 10)

    chunkedbaselinesstrings = ['\n\n'.join(x) for x in chunkedbaselines]

    #Go over each chunk separately
    for idx, sentence in enumerate(chunkedbaselinesstrings):
        #See if this chunk has been translated already
        if os.path.isfile(currentpath + '/Originallines_text2rdf.pkl'):
            with open(currentpath + '/Originallines_text2rdf.pkl', 'rb') as f:
                checklines = pickle.load(f)
            if chunkedbaselines[idx] in checklines:
                continue

        # Define text to translate
        text_to_translate = sentence

        # Start a Selenium driver
        chromedriver = currentpath + '/chromedriver.exe'

        driver = webdriver.Chrome(chromedriver)

        # Reach the deepL website
        deepl_url = 'https://www.deepl.com/en/translator'
        driver.get(deepl_url)

        # Go to the language selector and click on it
        button_css = 'button.lmt__language_select__active'
        button = driver.find_element_by_css_selector(button_css)
        button.click()
        time.sleep(1)

        #And select russian
        button_css = '//*[@id="dl_translator"]/div[3]/div[2]/div[1]/div[1]/div/div/button[10]'
        button = driver.find_element_by_xpath(button_css)
        button.click()
        time.sleep(1)

        #Go to the input area
        input_css = 'div.lmt__inner_textarea_container textarea'
        input_area = driver.find_element_by_css_selector(input_css)

        # Send the sentence to the translator
        input_area.clear()
        input_area.send_keys(text_to_translate)

        # Wait for translation to appear on the web page
        time.sleep(7)

        #Go to the cookie button and click on it
        button_css = 'button.dl_cookieBanner--buttonAll'
        button = driver.find_element_by_css_selector(button_css)
        button.click()

        time.sleep(1)

        # Go to the translate language and click on it
        button_css = '//*[@id="dl_translator"]/div[3]/div[2]/div[3]/div[1]/div[1]/div[1]/button'
        button = driver.find_element_by_xpath(button_css)
        button.click()

        time.sleep(1)

        # And select English
        button_css = '//*[@id="dl_translator"]/div[3]/div[2]/div[3]/div[1]/div[1]/div[1]/div/button[1]'
        button = driver.find_element_by_xpath(button_css)
        button.click()

        time.sleep(1)

        #Go to the copy button and click on it
        button_css = ' div.lmt__target_toolbar__copy button'
        button = driver.find_element_by_css_selector(button_css)
        button.click()

        time.sleep(2)

        # Quit selenium driver
        driver.quit()

        # Get content from clipboard
        content = pyperclip.paste()
        newcontent = re.split(r'\r\n\r\n', content)
        if newcontent[-1] == 'Translated with www.DeepL.com/Translator (free version)':
            newcontent = newcontent[:-1]

        # Display results
        print('_'*50)
        print('Original    :')
        print(text_to_translate.split('\n\n'))
        print('Translation :')
        print(newcontent)
        print('_'*50)
        if len(text_to_translate.split('\n\n')) != len(newcontent):
            print('Uneven division text and translation')
            exit(0)

        #Save the translated lines in one document
        if os.path.isfile(currentpath + '/Originallines_text2rdf.pkl'):
            with open(currentpath + '/Originallines_text2rdf.pkl', 'rb') as f:
                originallines = pickle.load(f)

            originallines.append(chunkedbaselines[idx])

            with open(currentpath + '/Originallines_text2rdf.pkl', 'wb') as f:
                pickle.dump(originallines, f)
        else:
            with open(currentpath + '/Originallines_text2rdf.pkl', 'wb') as f:
                pickle.dump([chunkedbaselines[idx]], f)

        if os.path.isfile(currentpath + '/Translatedlines_text2rdf.pkl'):
            with open(currentpath + '/Translatedlines_text2rdf.pkl', 'rb') as f:
                testlines = pickle.load(f)

            testlines = testlines + newcontent

            with open(currentpath + '/Translatedlines_text2rdf.pkl', 'wb') as f:
                pickle.dump(testlines, f)
        else:
            with open(currentpath + '/Translatedlines_text2rdf.pkl', 'wb') as f:
                pickle.dump(newcontent, f)

        pyperclip.copy(' ')

    with open(currentpath + '/Translatedlines_text2rdf.pkl', 'rb') as f:
        alllines = pickle.load(f)

    totaltranslatedlines = '\n'.join(alllines)

    with open(currentpath + '/' + outputname, 'wb') as f:
        f.write(bytes(totaltranslatedlines, 'UTF-8'))

if __name__ == '__main__':
    TranslateBaseline(sys.argv[1], sys.argv[2])
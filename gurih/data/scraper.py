from datetime import datetime
import time
import re
import os

import numpy as np
import pandas as pd

import requests
import urllib
from selenium import webdriver      
# from selenium.common.exceptions import NoSuchElementException
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

# from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
"""
tqdm.__version__ = '4.32.1'
"""

class BibleIsScraper:
    """
    BibleIsScraper(obj)

    Latest modified: January 17, 2020

    Scrape transcription and audio files from https://live.bible.is/bible/
    As page layout may dynamically change in the future, following methods may need adjustment:
        - get_urls
        - scrape_page

    Attributes
    ----------
    data : list of list
        Collection of scrape page data in following format:
            [url : str, chapter_string : str, audio_title : str]
        As the sraping process goes, this attributes will be updated accordingly by appending 
        the data list. Audio title filename is derived from url, describing chapter and verse 
        it points to.
            url            = https://live.bible.is/bible/INDASV/GEN/1?audio_type=audio 
            audio_filename = INDASV_GEN_1.mp3
    urls : list of str
        ALl chapter urls from bible.is to be used in scrape_all method. It is an empty list 
        until get_urls method is called.

    Parameters
    ----------
    driver_path : str
        chromedriver.exe filepath
    """
    def __init__(self, base_url, driver_path, output_dir='../../dataset/raw/bibleis/'):
        self.base_url = base_url
        self.driver_path = driver_path
        self.output_dir = output_dir
        self.data = []
        self.urls = []

        if not os.path.exists(self.output_dir): 
            os.makedirs(self.output_dir)
            print("Output directory created at " + self.output_dir)
        else:
            print("Output directory is already created at " + self.output_dir)
            
    def get_urls(self):
        """
        Method to get base urls for all chapters. You may need to update this method according
        to the latest page layout. You may use chrome devtools via inspect element to see the
        page source.

        Latest modified: January 17, 2020
        """
        base_url = self.base_url
        driver = webdriver.Chrome('./chromedriver.exe')
        driver.get(base_url)

        # The chapter url is hidden until after we click on the dropdown menu
        button =  driver.find_element_by_id("chapter-dropdown-button")
        driver.execute_script("arguments[0].click()", button)
        time.sleep(3)

        chapters = driver.find_elements_by_css_selector(".chapter-box")
        chapter_urls = [chapter.get_attribute('href') for chapter in chapters]
        
        # Apparently, chapter_urls will have None values for the base url, we need to replace
        # this None value to the base url
        none_index = chapter_urls.index(None)
        chapter_urls[none_index] = base_url

        # Save urls as class attribute
        self.urls = chapter_urls
        
        # Close the driver
        driver.close()
        
    def scrape_all(self, urls=None):
        if (not urls): 
            urls = self.urls
        
        for url in tqdm(urls):
            new_data = self.scrape_page(url)
            self.data.append(new_data)
            
    def scrape_page(self, url):
        """
        Methods to get transcription and download audio file from each url. You may need to 
        update this method according to the latest page layout. You may use chrome devtools 
        via inspect element to see the page source.

        Latest modified: January 17, 2020
        """
        driver = webdriver.Chrome(self.driver_path)
        driver.get(url)

        # Get all verse
        chapter_section = driver.find_element_by_css_selector(".chapter")
        verses          = chapter_section.find_elements_by_css_selector(".v")
        list_verse      = [verse.get_attribute("innerHTML") for verse in verses]
        chapter_string  = '\n\n'.join(list_verse)

        # Get audio file attributes
        audio       = driver.find_element_by_css_selector(".audio-player")
        audio_src   = audio.get_attribute("src")
        audio_title = re.search("[^?]*", url[28:]).group() + ".mp3"
        audio_title = audio_title.replace("/", "_")
        audio_title = self.output_dir + audio_title
        response    = urllib.request.urlopen(audio_src)

        with open(audio_title, "wb") as f:
            f.write(response.read())
            
            # Close the driver
            driver.close()

        return [url, chapter_string, audio_title]
    
    def to_dataframe(self):
        return pd.DataFrame(self.data, columns=["url", "chapter_string", "audio_title"])
    
    def write_csv(self, filename=None):
        if filename is None:
            filename = self.output_dir + 'bibleis_transcription.csv'
        self.to_dataframe().to_csv(filename, sep=',', line_terminator='\n', index=False)
        print("Data written in "+filename)
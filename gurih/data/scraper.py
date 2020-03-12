import os
import re
import time

import pandas as pd
import urllib
from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
from tqdm.auto import tqdm


class BibleIsScraper:
    """
    BibleIsScraper(obj)

    Latest modified: February 3, 2020

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

    Example
    -------
    If using get_urls for getting all urls:
    >>> scraper = BibleisScraper()
    >>> scraper.get_urls()
    >>> scraper.run()

    if providing your own urls:
    >>> urls = ["https://url_one", "https://url_two"]
    >>> scraper = BibleisScraper()
    >>> scraper.run(urls)

    To write the output:
    >>> len(scraper.data)
    2
    >>> scraper.write_csv("bibleis_transcription.csv")
    """
    def __init__(self, base_url, driver_path, output_dir='../../dataset/raw/bibleis/'):
        self.base_url = base_url
        self.driver_path = driver_path
        self.output_dir = output_dir if output_dir[-1] == '/' else output_dir + '/'
        self.data = []
        self.urls = []
        self.scrape_text = True
        self.scrape_audio = True
        self.debug = []

        # Get inferred version from base_url
        if 'INDASV' in base_url:
            version = 'INDASV'
        elif 'INDWBT' in base_url:
            version = 'INDWBT'
        else:
            raise ValueError("Base url version not supported."
                             " Required either INDASV or INDWBT version")
        self.version = version

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print("Output directory created at " + self.output_dir + "\n")
        else:
            print("Output directory is already created at " + self.output_dir + "\n")

        print("Scrape text: " + str(self.scrape_text))
        print("Scrape audio: " + str(self.scrape_audio))
        print("Edit the configuration by setting corresponding attributes.")

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
        button = driver.find_element_by_id("chapter-dropdown-button")
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

    def run(self, urls=None):
        if not urls:
            if not self.urls:
                raise AttributeError("Urls not defined. Use get_urls() or provide urls parameter.")
            else:
                urls = self.urls

        print("Running scraper:")
        print("Scrape text: " + str(self.scrape_text))
        print("Scrape audio: " + str(self.scrape_audio))

        if (self.scrape_audio or self.scrape_text) is False:
            raise AttributeError("Either self.scrape_audio or self.scrape_text should be True.")
        else:
            for url in tqdm(urls):
                try:
                    new_data = self.scrape_page(url)
                    self.data.append(new_data)
                except Exception as e:
                    print(url)
                    print(e)

    def scrape_page(self, url):
        """
        Methods to get transcription and download audio file from each url. You may need to
        update this method according to the latest page layout. You may use chrome devtools
        via inspect element to see the page source.

        Latest modified: January 17, 2020
        """
        driver = webdriver.Chrome(self.driver_path)
        driver.get(url)

        audio_title = ''
        chapter_string = ''

        if self.scrape_text:
            if self.version == 'INDASV':
                chapter_string = self._scrape_text_indasv(driver, url)
            elif self.version == 'INDWBT':
                chapter_string = self._scrape_text_indwbt(driver, url)

        if self.scrape_audio:
            # Get audio file attributes
            audio       = driver.find_element_by_css_selector(".audio-player")
            audio_src   = audio.get_attribute("src")
            audio_title = re.search("[^?]*", url[28:]).group() + ".mp3"
            audio_title = audio_title.replace("/", "_")
            audio_title = self.output_dir + "audio/" + audio_title
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
            filename = self.output_dir + 'transcript.csv'

        df = self.to_dataframe()
        self._check_null_df(df)

        self.to_dataframe().to_csv(filename, sep=',', line_terminator='\n', index=False)
        print("Data written in " + filename)

    def _check_null_df(self, df):
        """
        Check for null values in dataframe before writing it to .csv

        Parameters
        ----------
        df : pandas.DataFrame(columns=["url", "chapter_string", "audio_title"])
            dataframe of scraped data
        Raises
        ------
        ValueError
            if dataframe contain null values in either text or audio column
        """
        if self.scrape_text is True:
            sum_null_text = df['chapter_string'].isnull().sum()
            if sum_null_text > 0:
                raise ValueError(f"Found {sum_null_text} null values in chapter_string column.")
        if self.scrape_audio is True:
            sum_null_audio = df['audio_title'].isnull().sum()
            if sum_null_audio > 0:
                raise ValueError(f"Found {sum_null_audio} null values in audio_title column.")

    def _scrape_text_indasv(self, driver, url):
        chapter_string = ''

        # Get all verses
        cv = self.__get_chapter(url)
        chapter_section = driver.find_element_by_css_selector(".chapter")
        css_pattern = f"p[data-id^={cv}], span[data-id^={cv}], div[data-id^={cv}]"
        data = chapter_section.find_elements_by_css_selector(css_pattern)

        verses = []
        for d in data:
            d_text = d.get_attribute("innerHTML")
            idx = d_text.find('<')  # get all innerHTML until the first '<'
            if idx != 0:
                d_text = d_text[:idx]
            verses.extend([d_text])

        chapter_string = '\n\n'.join(verses)

        # Clean <span> with class="note"
        chapter_string = re.sub('<span class="note".+span>', '', chapter_string)

        return chapter_string

    def _scrape_text_indasv_old(self, driver, url):
        chapter_string = ''

        # Get all verses
        chapter_section = driver.find_element_by_css_selector(".chapter")
        ps = chapter_section.find_elements_by_css_selector("p")

        verses = []
        if len(ps) != 0:
            for p in ps:  # not including the chapter number
                p_text = p.get_attribute("innerHTML")  # get all text

                # Find disconnected verse, join it
                hanging_verse_idx = p_text.find('<')
                if hanging_verse_idx != 0:
                    hanging_verse = p_text[:hanging_verse_idx]
                    self.debug.append(f"{url} {hanging_verse}")
                    if len(verses) == 0:  # handle occurence in first <p>
                        verses.append(hanging_verse)
                    else:
                        last_verse = verses.pop()
                        verses.append(last_verse + " " + hanging_verse)

                other_verses = p.find_elements_by_css_selector(".v")
                other_verses = [v.get_attribute("innerHTML") for v in other_verses]
                verses.extend(other_verses)
        # handle chapter not having any p element
        else:
            other_verses = chapter_section.find_elements_by_css_selector(".v")
            other_verses = [v.get_attribute("innerHTML") for v in other_verses]
            verses.extend(other_verses)

        chapter_string = '\n\n'.join(verses)

        return chapter_string

    def _scrape_text_indwbt(self, driver, url):
        chapter_string = ''

        # Get all verses
        cv = self.__get_chapter(url)
        chapter_section = driver.find_element_by_css_selector(".chapter")
        css_pattern = f"p[data-id^={cv}], span[data-id^={cv}], div[data-id^={cv}]"
        data = chapter_section.find_elements_by_css_selector(css_pattern)

        verses = []
        for d in data:
            d_text = d.get_attribute("innerHTML")
            idx = d_text.find('<')  # get all innerHTML until the first '<'
            if idx != 0:
                d_text = d_text[:idx]
            verses.extend([d_text])

        chapter_string = '\n\n'.join(verses)

        # Clean <span> with class="note"
        chapter_string = re.sub('<span class="note".+span>', '', chapter_string)

        return chapter_string

    @staticmethod
    def __get_chapter(url):
        """
        "https://live.bible.is/bible/INDWBT/MAT/1?audio_type=audio" --> "MAT1"
        """
        cv_pattern = re.search("[^?]*", url[35:]).group().replace("/", '')

        # In case of starting with digit, WTF??
        if cv_pattern[0] in ['1', '2', '3']:
            cv_pattern = r'\3' + cv_pattern[0] + ' ' + cv_pattern[1:]

        return cv_pattern

import unittest

from gurih.data.scraper import BibleIsScraper

class BibleisScraperTest(unittest.TestCase):
    """Test suite for BibleIsScraper class"""
    @classmethod
    def setUpClass(cls):
        base_url = "https://live.bible.is/bible/INDASV/MRK/1?audio_type=audio"
        cls._scraper = BibleIsScraper(base_url, "chromedriver.exe")
        
    @classmethod
    def tearDownClass(cls):
        del cls._scraper

    def test_scrape_page(self):
        """
        TODO: configure out how to do test on server
        """
        pass

if __name__ == "__main__":
    unittest.main()
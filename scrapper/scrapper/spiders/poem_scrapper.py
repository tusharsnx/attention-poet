import scrapy
from scrapy.selector import Selector
import json
import time

class PoemSpider(scrapy.Spider):
    name = "poemspider"
    start_urls = ["https://www.amarujala.com/kavya/mere-alfaz/savita-sharma-chaha-jisko-humne-woh-mila-hi-nahi?src=rcmd"]
        
    def __init__(self):
        self.base_url = "https://www.amarujala.com"
        self.feed_url = "https://www.amarujala.com/ajax/getKavyaAjaxData"
        self.feeds_offset = 0  # multiple of 10
        self.feeds_end = 80     # multiple of 10
        self.parsed_poems = 0
        self.skipped_poems = 0

        # to check if site already scraped including previous crawls
        self.visited_urls = set()
        with open("scrapped.json", "r") as f:
            data = f.read()
            if data:
                data = json.loads(data)

        for item in data:
            try:
                self.visited_urls.add(item["url"])
            except:
                pass
    

    def parse(self, response, feeds=False):
        assert self.feeds_offset%10==0, "feeds_offset must be multiple of 10"
        assert self.feeds_end%10==0, "feeds_end must be multiple of 10"
        
        print("\n", "current url: ", response.url, "\n")

        self.parsed_poems+=1
        data, story_id = self.extract_peom(response)
        
        if response.url not in self.visited_urls:
            self.visited_urls.add(response.url)
            yield data


        # parsing additional peoms from feeds
        if not feeds:
            for i in range((self.feeds_end-self.feeds_offset)//10):
                query_params = self.get_query_params(story_id=story_id, skip=self.feeds_offset+(i+1)*10)
                yield scrapy.Request(
                    method="GET", url=self.feed_url+"?"+query_params, 
                    callback=self.parse_feed
                )
                # print("feeds parsed: ", (i+1)*10)
        print("total parsed peoms", self.parsed_poems)
        print("total skipped peoms", self.skipped_poems)

    
    def extract_peom(self, response):
        poem = response.css(".kavya_article")
        story_id = response.css(".feed-scroll::attr(data-storyid)").get()
        
        if poem.css("#slide-2").get() is not None:
            # print("has slides")
            lines = []
            first_slide = poem.css("pre p::text").getall()
            for line in first_slide:
                if len(line)>=300:
                    continue
                else:
                    lines.append(self.clean_text(line))
            pre_divs = poem.css("pre > div")
            for item in pre_divs:
                if item.css("::attr(id)").get().startswith("slide"):
                    ps = item.css(".desc p::text").getall()
                    for line in ps:
                        if len(line)>=300:
                            continue
                        else:
                            lines.append(self.clean_text(line))
            data = {
                "url": response.url,
                "lines": lines,
                } 
        
        else:
            # print("no slides")
            lines = []
            ps = poem.css("pre::text").getall()
            for line in ps:
                if len(line)>300:
                    continue
                else:
                    lines.append(self.clean_text(line))
            data = {
                "url": response.url,
                "lines": lines,
                }
            
        return data, story_id
 
        
    def parse_feed(self, response):
        raw_data = response.body
        data = json.loads(raw_data)
        html_data = data["kavyaList"]
        response = Selector(text=html_data)
        for link in response.css(".kavya-read-more::attr(href)"):
            # print("link: ", link.get())
            full_link = self.base_url+link.get()
            if full_link not in self.visited_urls:
                yield scrapy.Request(url=full_link, callback=self.parse, cb_kwargs=dict(feeds=True))
            else:
                print("Skipped........")
                self.skipped_poems+=1
    
    
    # create query params for the request
    @staticmethod
    def get_query_params(story_id, skip=0):
        v = int(time.time())
        slug = "kavya"
        return f"v={v}&skip={skip}&slug={slug}&story_id={story_id}"

    @staticmethod
    def clean_text(text):
        return text.replace("\n", "").strip()
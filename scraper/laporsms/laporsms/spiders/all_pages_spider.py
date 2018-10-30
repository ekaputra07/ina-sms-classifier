import scrapy

class AllPagesSpider(scrapy.Spider):
  name = "all_pages"

  def start_requests(self):
    url = "http://laporsms.com/laporan-masyarakat/index.php?r=site/index&Sms_page=778"
    yield scrapy.Request(url=url, callback=self.parse)

  def parse(self, response):    

    rows = response.css("tr")
    for r in rows:
      cols = r.css("td::text").extract()
      if len(cols) == 5:
        yield {
          "type": cols[0],
          "message": cols[1],
          "submitted": cols[2],
          "received": cols[3],
          "sender": cols[4]
        }

    next_container = response.css("li.next").extract_first()
    if not "hidden" in next_container:
      next_url = response.css("li.next a::attr(href)").extract_first()
      yield response.follow(next_url, callback=self.parse)
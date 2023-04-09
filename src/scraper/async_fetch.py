import asyncio
from bs4 import BeautifulSoup
from timeit import default_timer
from aiohttp import ClientSession
import random

class FetchAsync:

    def __init__(self, urls) -> None:
        self.urls = urls
        self.responses = {}

    def get_soups(self):
        soups = {}
        for url in self.responses:
            soup = BeautifulSoup(self.responses[url].decode('utf-8'), features='lxml')
            soups[url.split('=')[1]] = soup
        return soups


    async def fetch(self, url, session):
        self.start_time[url] = default_timer()
        choices =[4,6,8,10,12,14,16,18,20,22]
        choice = random.choice(choices)
        await asyncio.sleep(choice)
        async with session.get(url, headers={'User-Agent': 'Mozilla/5.0'}) as response:
            resp = await response.read()
            elapsed = default_timer() - self.start_time[url]
            print(url   , ' took ',   str(elapsed), choice)
            self.responses[url] = resp
            return resp

    async def fetch_all(self, urls):
        tasks = []
        self.start_time = dict()
        async with ClientSession() as session:
            for url in urls:
                task = asyncio.ensure_future(self.fetch(url, session))
                tasks.append(task)
            _ = await asyncio.gather(*tasks)

    def fetch_async(self):
        start_time = default_timer()

        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(self.fetch_all(self.urls))
        loop.run_until_complete(future)

        tot_elapsed = default_timer() - start_time

        print('Total time taken : ',  str(tot_elapsed))




if __name__ == '__main__':
    urls = ['https://nytimes.com',
                'https://github.com',
                'https://google.com',
                'https://reddit.com',
                'https://producthunt.com']

    urls = [f'https://finviz.com/quote.ashx?t={ticker}' for ticker in ['AAPL', 'F', 'T']]

    fa = FetchAsync(urls=urls)
    fa.fetch_async()
    print(fa.get_soups())

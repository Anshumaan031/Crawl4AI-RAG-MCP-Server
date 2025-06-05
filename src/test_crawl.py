import asyncio
import platform
from crawl4ai import AsyncWebCrawler, BrowserConfig

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

async def test_crawl():
    config = BrowserConfig(
        headless=True,
        browser_type="chromium",
        extra_args=["--no-sandbox", "--disable-dev-shm-usage"]
    )
    
    async with AsyncWebCrawler(config=config) as crawler:
        result = await crawler.arun(url="https://www.dsci.in/")
        print(f"Success: {result.success}")
        print(f"Content length: {len(result.markdown) if result.markdown else 0}")

if __name__ == "__main__":
    asyncio.run(test_crawl())
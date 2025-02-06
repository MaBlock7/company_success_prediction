# Text-based Company Success Prediction documentation!

## Commands

The Makefile contains the central entry points for common tasks related to this project.

## Introduction

This project explores the usefulness of company website data to evaluate early startup success when only limited financial data is available.

## Methods

### Company URL Scraping

First, we use [Bright Data's](https://brightdata.com/) SERP api to scrape company website URL's from Google Maps by creating a search string using the exact company name and town from the Swiss Business Registry. Furthermore, we apply the following additional parameters:

- gl=ch to get search results for Switzerland.
- hl=de to get search results in German.
- num=5 to get at most 5 matches per search.
- brd_json=1 to get processed json data back for each search.

To match potential URL's from the search results to companies, we apply the following heuristics in order of importance. A URL belongs to a company if:

1. The company name from the business registry matches the matched name exactly and either the town or the street name appear in the address,
2. The matched name is only a partial match (e.g. 'bolz solutions' -> 'bolz solutions gmbh') but the street name matches perfectly,
3. The company name from the business registry matches the matched name exactly, but the address data is missing on Google Maps.

These heuristigs allow us to create an initial mapping of potential URL to company matches, these matches are then further processed to ensure, only true matches remain in the final dataset.

### Company Website Scraping

To scrape the content of a company website, we use the open-source library [crawl4ai](https://crawl4ai.com/mkdocs/), which provides webscraping and content extraction tools optimized for the use of RAG and LLM systems. Based on the URL we fetched from Google Maps, we first try to search for the sitemap of a the website. We extract every URL from all sitemaps we find. If there is no sitemap, we scrape all the internal links from the homepage (base URL).

The extracted URLs are then filtered on a set of simple heuristics which include 1) removing URLs that have a language indicator other than en (English) or de (German), 2) removing URLs that contain an indicator for pages that are of no interest for our purpose, such as 'news', 'terms-of-use', or 'impressum'.

For each of the remaining URLs, we scrape the content and extract it as Markdown optimized for LLMs and the creation of subsequent embeddings. Since many Startups in Switzerland only provide webpages in English, we prioritize English content by passing the Accept-Language parameter to the header, and add German as the second option. In the rare case, where a webpage does neither offer a version in English nor German, we crawl the Content in its original Language. However, By relying on Multi-lingual embeddings, we nevertheless allows us to cluster them, even if the content might be in different languages.

### Retriever



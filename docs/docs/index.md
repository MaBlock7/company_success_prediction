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

...

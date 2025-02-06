import json
import ssl
import urllib.request
import urllib.parse

ssl._create_default_https_context = ssl._create_unverified_context


class GoogleMapsScraper:
    """A scraper for retrieving business listings from Google Maps search results.

    This scraper constructs search queries for Google Maps based on company names and locations,
    fetches results via a proxy, and optionally returns structured JSON output.

    Features:
    - Constructs Google Maps search URLs with language and region parameters.
    - Uses a proxy for fetching search results.
    - Returns JSON-formatted results if requested.

    Args:
        auth_token (str): Authentication token for accessing the proxy service.
        country_of_search (str, optional): The country code for search results (default: 'ch' for Switzerland).
        page_language (str, optional): The language for search results (default: 'de' for German).
        num_results (int, optional): The number of search results to return (default: 5).
        json_output (int, optional): Whether to return results in JSON format (default: 1).

    Example:
        >>> scraper = GoogleMapsScraper(auth_token="your_token")
        >>> results = scraper.search("Example Solutions AG", "Bern")
    """
    def __init__(
        self,
        auth_token: str,
        country_of_search: str = 'ch',
        page_language: str = 'de',
        num_results: int = 5,
        json_output: int = 1
    ):
        gl_param = f'gl={country_of_search}' if country_of_search else None
        hl_param = f'hl={page_language}' if page_language else None
        num_param = f'num={num_results}' if num_results else None
        self.json_output = json_output
        brd_json_param = f'brd_json={json_output}' if json_output else None

        self.search_parameters = '&'.join([param for param in [gl_param, hl_param, num_param, brd_json_param] if param])
        self.base_url = 'https://www.google.com/maps/search/{company_name}+{town}/?{search_parameters}'

        proxy_url = f'http://brd-customer-hl_15a41d55-zone-serp_api1:{auth_token}@brd.superproxy.io:33335'
        self.opener = self._define_opener(proxy_url)

    def _create_maps_url(self, company_name: str, town: str) -> str:
        """Constructs a Google Maps search URL with the given company name and town."""
        company_name = self._prepare_search_queries(company_name)
        town = self._prepare_search_queries(town)
        return self.base_url.format(
            search_parameters=self.search_parameters,
            company_name=company_name,
            town=town,
        )

    @staticmethod
    def _prepare_search_queries(string: str) -> str:
        """Formats a search query by converting spaces to '+' and encoding special characters."""
        return urllib.parse.quote(string.lower().strip().replace(' ', '+'), safe='+')

    @staticmethod
    def _define_opener(proxy_url: str) -> urllib.request.OpenerDirector:
        """Creates an HTTP request opener with a configured proxy."""
        return urllib.request.build_opener(urllib.request.ProxyHandler({'http': proxy_url, 'https': proxy_url}))

    def search(self, company_name: str, town: str) -> dict | str:
        """Performs a search on Google Maps for the specified company and location.

        Args:
            company_name (str): The name of the company to search for.
            town (str): The town or city where the company is located.

        Returns:
            dict | str: The search results in JSON format if `json_output` is enabled, otherwise a raw HTML response.
        """
        search_url = self._create_maps_url(company_name, town)
        output = self.opener.open(search_url).read()
        return json.loads(output) if self.json_output else output

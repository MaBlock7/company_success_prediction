import json
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import urllib.request


class GoogleMapsScraper:
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
        self.base_url = 'https://www.google.com/maps/search/{company_name}+{town}+{street}/?{search_parameters}'

        proxy_url = f'http://brd-customer-hl_15a41d55-zone-serp_api1:{auth_token}@brd.superproxy.io:33335'
        self.opener = self._define_opener(proxy_url)

    def _create_maps_url(self, company_name: str, town: str, street: str) -> str:
        """
        """
        company_name = self._prepare_search_queries(company_name)
        town = self._prepare_search_queries(town)
        street = self._prepare_search_queries(street)
        return self.base_url.format(
            search_parameters=self.search_parameters,
            company_name=company_name,
            town=town,
            street=street,
        )

    @staticmethod
    def _prepare_search_queries(string: str) -> str:
        return string.lower().strip().replace(' ', '+')

    @staticmethod
    def _define_opener(proxy_url: str):
        """
        """
        return urllib.request.build_opener(urllib.request.ProxyHandler({'http': proxy_url, 'https': proxy_url}))

    def search(self, company_name: str, town: str, street: str) -> ...:
        """
        """
        search_url = self._create_maps_url(company_name, town, street)
        output = self.opener.open(search_url).read()
        return json.loads(output) if self.json_output else output

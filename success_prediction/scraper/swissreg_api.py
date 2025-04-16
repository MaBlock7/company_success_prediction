import requests
import xml.etree.ElementTree as ET
from time import sleep
from typing import Literal, Optional, Generator


class SwissregCrawler:
    BASE_URL = "https://www.swissreg.ch/servlet/servlet.dataDelivery"
    NAMESPACES = {
        'api': 'urn:ige:schema:xsd:datadeliverycore-1.0.0',
        'tmk': 'urn:ige:schema:xsd:datadeliverytrademark-1.0.0',
        'ptt': 'urn:ige:schema:xsd:datadeliverypatent-1.0.0',
    }

    def __init__(self, endpoint: Literal["TrademarkSearch", "PatentSearch"] = "TrademarkSearch", page_size: int = 64):
        self.endpoint = endpoint
        self.page_size = page_size
        self.root_tag = 'tmk' if endpoint == "TrademarkSearch" else 'ptt'

    def _create_initial_request(self) -> str:
        return f"""<?xml version="1.0"?>
<ApiRequest xmlns="urn:ige:schema:xsd:datadeliverycore-1.0.0"
            xmlns:{self.root_tag}="urn:ige:schema:xsd:datadelivery{self.endpoint.lower().replace("search", "")}-1.0.0">
  <Action type="{self.endpoint}">
    <{self.root_tag}:{self.endpoint}Request xmlns="urn:ige:schema:xsd:datadeliverycommon-1.0.0">
      <Page size="{self.page_size}"/>
      <Query>
        <LastUpdate/>
      </Query>
      <Sort>
        <LastUpdateSort>Ascending</LastUpdateSort>
      </Sort>
    </{self.root_tag}:{self.endpoint}Request>
  </Action>
</ApiRequest>
"""

    def _create_continuation_request(self, continuation_token: str) -> str:
        return f"""<?xml version="1.0"?>
<ApiRequest xmlns="urn:ige:schema:xsd:datadeliverycore-1.0.0">
  <Continuation>{continuation_token}</Continuation>
</ApiRequest>
"""

    def _post_request(self, xml_payload: str) -> ET.Element:
        headers = {'Content-Type': 'application/xml'}
        response = requests.post(self.BASE_URL, data=xml_payload.encode('utf-8'), headers=headers)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 10))
            sleep(retry_after)
            return self._post_request(xml_payload)
        response.raise_for_status()
        return ET.fromstring(response.content)

    def _parse_items(self, root: ET.Element) -> list:
        return root.findall('.//api:Item', self.NAMESPACES)

    def _get_continuation_token(self, root: ET.Element) -> Optional[str]:
        continuation = root.find('.//api:Continuation', self.NAMESPACES)
        return continuation.text if continuation is not None else None

    def traverse(self) -> Generator[ET.Element, None, None]:
        xml_payload = self._create_initial_request()
        while True:
            root = self._post_request(xml_payload)
            items = self._parse_items(root)
            for item in items:
                yield item
            continuation_token = self._get_continuation_token(root)
            if not continuation_token:
                break
            xml_payload = self._create_continuation_request(continuation_token)

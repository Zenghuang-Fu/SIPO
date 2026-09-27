"""Search tool backed by the bundled local retrieval service."""
import http.client
import json
import logging
from typing import Dict
from .base_tool import BaseTool


class LocalSearchTool(BaseTool):
    def __init__(self, host="localhost", port=8000, max_results=10,
                 result_length=1000, topk=3):
        if not host or "/" in host or "@" in host:
            raise ValueError("host must be a hostname or IP address")
        if not 1 <= int(port) <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if min(max_results, result_length, topk) < 1:
            raise ValueError("result limits must be positive")
        self.host, self.port = host, int(port)
        self._max_results, self._result_length = max_results, result_length
        self.topk = topk

    @property
    def name(self):
        return "local_search"

    @property
    def trigger_tag(self):
        return "search"

    def execute(self, query: str, timeout: int = 60) -> str:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            payload = {"queries": [query.replace('"', '')],
                       "topk": self.topk, "return_scores": True}
            connection.request("POST", "/retrieve", body=json.dumps(payload),
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f"retrieval returned HTTP {response.status}")
            return self._extract_and_format_results(json.loads(response.read()))
        except Exception as exc:
            logging.getLogger(__name__).warning("Search failed (%s)", type(exc).__name__)
            return ""
        finally:
            connection.close()

    def _extract_and_format_results(self, data: Dict) -> str:
        """
        Extract and format search results from API response.
        
        Args:
            data: API response data
            
        Returns:
            Formatted search results as string
        """
        # If no organic results, return empty response
        # if 'organic' not in data:
        #     data['chunk_content'] = []
        #     return self._format_results(data)

        # Extract unique snippets
        chunk_content_list = []
        seen_snippets = set()
        # for result in data['organic']:
        #     snippet = result.get('description', '').strip()
        #     if len(snippet) > 0 and snippet not in seen_snippets:
        #         chunk_content_list.append(snippet)
        #         seen_snippets.add(snippet)
        for result_group in data['result']:  # 第一层 list
            for result in result_group:       # 第二层 list
                document = result.get('document', {}) 
                snippet = document.get('contents', '').strip()

                if snippet and snippet not in seen_snippets:
                    chunk_content_list.append(snippet)
                    seen_snippets.add(snippet)

        data['chunk_content'] = chunk_content_list
        return self._format_results(data)

    def _format_results(self, results: Dict) -> str:
        """
        Format search results into readable text.
        
        Args:
            results: Dictionary containing search results
            
        Returns:
            Formatted string of search results
        """
        if not results.get("chunk_content"):
            return "No search results found."

        formatted = []
        for idx, snippet in enumerate(results["chunk_content"][:self._max_results], 1):
            snippet = snippet[:self._result_length]
            formatted.append(f"Page {idx}: {snippet}")
        
        return "\n".join(formatted)


# Compatibility for existing tool class imports.
BingSearchTool = LocalSearchTool

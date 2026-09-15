"""Provider adapters for VLM and embedding APIs.

Architecture
------------
Each capability (VLM, embedding) defines an abstract base class. Concrete
providers (Anthropic, OpenAI, Google) inherit from it. A factory function
selects the active provider based on environment variables.

Students: study this pattern. The same shape is what we expect for your own
extension points (e.g. storage backends, news sources, etc.).
"""
